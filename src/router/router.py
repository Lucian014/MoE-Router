"""
router/router.py -- RabbitMQ consumer that picks a model for every prompt.

    Gateway --router.requests--> Router --worker.<model>--> Worker --reply_to--> Gateway

For each request:
  1. with a session_id: read the session's routing state and its last turns from Postgres (R1)
  2. in parallel:
       - domain + complexity (session_routing.py): two independent Laya classifiers
         (classifier.py) on the current prompt -- a long one as its request + the start of the
         document (classifier_input.py) -- or the session's last routing, for a short /
         ambiguous follow-up
       - retrieval (RAG_VARIANT, V1 / V2; docs/rag-planv2.md 4.4-4.6): the query embedded with
         the chunks' model, every earlier chunk of the session with its distance (R2), dense +
         BM25 ranking, RRF, dedup -> the pool, with its texts (R3)
  3. deterministic rule over the catalog (routing_rule.py):
       enabled models of that category -> the one on the predicted tier -> else the other tier
       -> no model in the category: the is_default model
  4. the context in that model's token budget (shared/memory/assembly.py): system prompt, the
     session's recent turns, the top-k fragments of the turns that are not included whole (and of
     the last turn, when it had to be cut), the prompt
  5. forward to worker.<model> keeping correlation_id + reply_to (the worker answers the Gateway)

A retrieval failure (the embedding model unreachable, ...) doesn't fail the request: it is
answered with the recent turns only and recorded as V0.

The catalog is read from Postgres at start-up and re-read every CATALOG_REFRESH_SECONDS, so
enabling / disabling / adding a model in the DB takes effect without restarting the router.

Lifecycle (shared/lifecycle.py): start-up waits (with backoff) for a valid catalog and for
RabbitMQ; the readiness file exists while the router is consuming; on SIGTERM it stops
consuming, finishes the in-flight message (SHUTDOWN_GRACE_SECONDS) and closes.
"""
import asyncio
import json
import sys
import time
import uuid
from dataclasses import dataclass, field

from database import (
    ChatSessionRepository,
    ChunkRepository,
    ModelRepository,
    TurnRepository,
    dispose_engine,
    get_session,
)
from router.classifier import CATEGORIES, load_classifiers, predict_complexity, predict_domain
from router.routing_rule import catalog_from_rows, default_model, select_model, validate_catalog
from router.session_routing import RoutingRules, SessionState, decide
from shared.config import (
    get_classifier_settings,
    get_lifecycle_settings,
    get_memory_settings,
    get_rabbitmq_settings,
    get_retrieval_settings,
    get_router_settings,
)
from shared.lifecycle import Stopped, install_stop_signals, retry, set_ready
from shared.memory.analyzer import analyze
from shared.memory.assembly import Budget, Fragment, TurnText, build_messages
from shared.memory.embedding import EmbeddingClient, OllamaEmbeddingClient
from shared.memory.header import build_dense_query
from shared.memory.presets import Preset, get_preset
from shared.memory.retrieval import RankParams, rank
from shared.prompts import SYSTEM_PROMPT
from shared.rabbitmq import RabbitMQClient, remaining_seconds
from shared.utils import model_to_queue

mq_client: RabbitMQClient | None = None
_embedder: EmbeddingClient | None = None

_CATALOG: list = []


# ------------------------------------------------------------------ catalog
def _signature(catalog):
    return sorted((m.model_id, m.enabled, m.complexity_tier, m.category, m.is_default,
                   m.context_window, m.max_output_tokens)
                  for m in catalog)


def _print_catalog(catalog, warnings):
    print(f"[OK] catalog: {sum(m.enabled for m in catalog)} enabled model(s), "
          f"default = {default_model(catalog).model_id}")
    for c in CATEGORIES:
        low, _ = select_model(catalog, c, "low")
        high, _ = select_model(catalog, c, "high")
        print(f"     {c:17s} low -> {low.model_id:32s} high -> {high.model_id}")
    for w in warnings:
        print(f"     [warn] {w}")


async def load_router_catalog() -> bool:
    """Read the `models` table, validate it and swap it in. Returns True if it changed.
    Raises on an invalid catalog (no / several default models, two models on the same
    category + tier): at start-up the router retries until the catalog is valid (and stays
    not ready meanwhile), during a refresh the previous catalog is kept."""
    global _CATALOG
    async with get_session() as session:
        entries = await ModelRepository(session).list_all()
    catalog = catalog_from_rows([e.to_dict() for e in entries])

    errors, warnings = validate_catalog(catalog, CATEGORIES)
    if errors:
        raise ValueError("invalid model catalog:\n  " + "\n  ".join(errors))
    changed = _signature(catalog) != _signature(_CATALOG)
    _CATALOG = catalog
    if changed:
        _print_catalog(catalog, warnings)
    return changed


async def refresh_catalog_forever() -> None:
    while True:
        await asyncio.sleep(get_router_settings().catalog_refresh_seconds)
        try:
            if await load_router_catalog():
                print("[*] catalog changed in the DB -> reloaded")
        except Exception as e:
            print(f"[!] catalog refresh failed, keeping the previous one: {e}")


# ------------------------------------------------------------------ routing
def classify(prompt: str):
    return predict_domain(prompt), predict_complexity(prompt)


def routing_rules() -> RoutingRules:
    cfg = get_router_settings()
    return RoutingRules(
        short_prompt_max_words=cfg.short_prompt_max_words,
        min_confidence=cfg.domain_min_confidence,
        min_margin=cfg.domain_min_margin,
        complexity_threshold=get_classifier_settings().complexity_threshold,
    )


async def load_session(session_id: uuid.UUID) -> tuple[SessionState | None, list[TurnText]]:
    """R1: the session's routing state and its last RAG_RECENT_TURNS_MAX turns (oldest first)."""
    async with get_session() as session:
        row = await ChatSessionRepository(session).get(session_id)
        turns = await TurnRepository(session).list_recent(session_id, get_retrieval_settings().recent_turns_max)
    state = SessionState(row.domain, row.domain_confidence, row.complexity_score) if row else None
    return state, [TurnText(t.turn, t.user_content, t.assistant_content) for t in turns]


# ------------------------------------------------------------------ retrieval
def resolve_variant(requested: str | None) -> Preset:
    """RAG_VARIANT, or the request's `variant` when ALLOW_VARIANT_OVERRIDE is on (evaluation).
    An unknown or offline-only variant asked for explicitly is an error, not a silent fallback."""
    cfg = get_retrieval_settings()
    if not requested or requested == cfg.variant:
        return get_preset(cfg.variant)
    if not cfg.allow_variant_override:
        print(f"[!] variant {requested!r} requested, but ALLOW_VARIANT_OVERRIDE is off: using {cfg.variant}")
        return get_preset(cfg.variant)
    preset = get_preset(requested)
    if not preset.online:
        raise ValueError(f"variant {requested} needs offline chunk context; the router runs V0-V2")
    return preset


def embedder() -> EmbeddingClient:
    global _embedder
    if _embedder is None:
        cfg = get_memory_settings()
        _embedder = OllamaEmbeddingClient(cfg.embedding_host, cfg.embedding_model,
                                          keep_alive=cfg.embedding_keep_alive, timeout=30.0)
    return _embedder


@dataclass
class Retrieved:
    fragments: list[Fragment]           # the pool, best first, with their texts
    candidates: int                     # chunks R2 returned
    timings: dict[str, float] = field(default_factory=dict)


def _ms(since: float) -> float:
    return round((time.perf_counter() - since) * 1000, 1)


async def retrieve(session_id: uuid.UUID, before_turn: int, prompt: str, preset: Preset) -> Retrieved:
    """Query -> R2 -> rank -> R3. Runs before the model is known: the pool is cut to top-k by
    build_messages, once the turns included whole are known."""
    cfg = get_retrieval_settings()
    timings: dict[str, float] = {}
    vector, model = None, None
    if preset.dense:
        t = time.perf_counter()
        client = embedder()
        (vector,) = await client.embed([build_dense_query(prompt, cfg.dense_query_max_tokens)])
        model = client.model
        timings["embed_ms"] = _ms(t)
    query_terms = analyze(prompt) if preset.bm25 else []

    async with get_session() as session:
        chunks = ChunkRepository(session)
        t = time.perf_counter()
        candidates = await chunks.list_candidates(session_id, before_turn, vector,
                                                  contextual=preset.contextual, embedding_model=model)
        timings["search_ms"] = _ms(t)
        t = time.perf_counter()
        pool = rank(candidates, query_terms, preset,
                    RankParams(candidates=cfg.candidates, pool=cfg.pool, dedup=cfg.dedup, jaccard=cfg.dedup_jaccard))
        timings["rank_ms"] = _ms(t)
        t = time.perf_counter()
        texts = await chunks.get_fragments([r.id for r in pool])
        timings["fetch_ms"] = _ms(t)

    fragments = [Fragment(r.id, f.turn, f.ord, f.content, f.header, f.header_from_answer)
                 for r in pool if (f := texts.get(r.id))]
    return Retrieved(fragments, len(candidates), timings)


def _r(x, digits=4):
    return None if x is None else round(x, digits)


async def route_message(message) -> None:
    try:
        payload = json.loads(message.body.decode())
        prompt = payload.get("prompt", "")
        if not prompt:
            raise ValueError("Payload is missing the required 'prompt' field.")

        preset = resolve_variant(payload.get("variant"))

        # 1. the session (none for a client that sends no session_id)
        state, turns, session_error, session_id = None, [], None, None
        if payload.get("session_id"):
            try:
                session_id = uuid.UUID(str(payload["session_id"]))
                state, turns = await load_session(session_id)
            except Exception as e:      # answer without history rather than not at all
                session_error = f"{type(e).__name__}: {e}"
                print(f"[!] {message.correlation_id}: session not loaded, routing without it: {session_error}")
        next_turn = (turns[-1].turn if turns else 0) + 1

        # 2. domain + complexity (the classifiers run only when the session can't answer), and
        #    retrieval at the same time -- from the first earlier turn: when the last turn is too
        #    long for the model, its own chunks stand in for the part that is cut
        async def route_on_session():
            t = time.perf_counter()
            return await asyncio.to_thread(decide, prompt, state, classify, routing_rules()), _ms(t)

        async def retrieve_or_error():
            try:
                return await retrieve(session_id, next_turn, prompt, preset)
            except Exception as e:
                return e

        use_rag = preset.retrieves and session_id is not None and session_error is None and bool(turns)
        if use_rag:
            (decision, classify_ms), retrieved = await asyncio.gather(route_on_session(), retrieve_or_error())
        else:
            (decision, classify_ms), retrieved = await route_on_session(), None
        rag_variant, fragments, retrieval_info = preset.name, [], None
        if isinstance(retrieved, Exception):
            rag_variant = "V0"
            retrieval_info = {"error": f"{type(retrieved).__name__}: {retrieved}"}
            print(f"[!] {message.correlation_id}: retrieval failed, recent turns only: {retrieval_info['error']}")
        elif retrieved is not None:
            fragments = retrieved.fragments
            retrieval_info = {"candidates": retrieved.candidates, "pool": len(fragments), **retrieved.timings}

        # 3. the model
        model, reason = select_model(_CATALOG, decision.domain, decision.complexity_level)
        selected_model = model.model_id
        worker_queue = model_to_queue(selected_model)

        # 4. the context, in that model's budget
        retrieval = get_retrieval_settings()
        context = build_messages(
            SYSTEM_PROMPT, turns, prompt,
            Budget(model.context_window, model.max_output_tokens, margin=retrieval.context_margin,
                   recent_max_share=retrieval.recent_max_share,
                   last_turn_max_share=retrieval.last_turn_max_share,
                   max_context_tokens=retrieval.max_context_tokens),
            fragments=fragments, top_k=retrieval.top_k, max_per_turn=retrieval.max_chunks_per_turn,
        )

        domain = decision.domain_prediction
        payload.update({
            "model": selected_model,
            "routing_reason": reason,                      # match | other-level | fallback:no-model
            # what the classifiers said (None when a follow-up skipped them)
            "task_type": decision.predicted_domain,
            "task_type_confidence": _r(domain.confidence) if domain else None,
            "task_type_probabilities": ({k: round(v, 4) for k, v in domain.probabilities.items()}
                                        if domain else None),
            "complexity_score": _r(decision.predicted_complexity),
            # full | edges (a long prompt, given as its request + the document's start) | cut
            "classifier_input": getattr(domain, "input", None) if domain else None,
            # what the request was routed on
            "session_routing": decision.reason,            # first-turn | classified | short-prompt | ambiguous-domain
            "inherited": decision.inherited,
            "effective_domain": decision.domain,
            "effective_domain_confidence": _r(decision.domain_confidence),
            "effective_complexity": _r(decision.complexity_score),
            "complexity": decision.complexity_level,
            "classify_ms": classify_ms,
            # the context
            "messages": context.messages,
            "num_ctx": model.context_window,
            "max_tokens": model.max_output_tokens,
            "rag_variant": rag_variant,
            "recent_turns": context.recent_turns,
            "retrieved_chunk_ids": context.fragment_ids,
            "retrieval": retrieval_info,               # candidates, pool, embed/search/rank/fetch ms
            "context_tokens": context.input_tokens,
        })
        if session_id is not None and session_error is None:
            payload["turn"] = next_turn                # 1 for a new session

        print(f"[*] {message.correlation_id}: {decision.reason}"
              f"{' (inherited)' if decision.inherited else ''} domain={decision.domain} "
              f"complexity={decision.complexity_level} -> {selected_model} [{reason}] -> {worker_queue} | "
              f"context [{rag_variant}]: {context.recent_turns}/{len(turns)} turns, "
              f"{len(context.fragment_ids)} fragments, ~{context.input_tokens} tokens"
              f"{', last turn cut' if context.last_turn_truncated else ''} ({classify_ms:.0f} ms)")

        # forward() preserves correlation_id + reply_to (+ the deadline)
        if not await mq_client.forward(message=message, routing_key=worker_queue, payload=payload):
            print(f"[!] {message.correlation_id}: deadline passed during routing, dropped")

    except json.JSONDecodeError:
        print("[!] Received invalid JSON")
        raise
    except Exception as e:
        print(f"[!] Error routing message: {e}")
        raise


async def main():
    global mq_client
    sys.stdout.reconfigure(line_buffering=True)
    rabbit = get_rabbitmq_settings()
    lifecycle = get_lifecycle_settings()
    refresh_seconds = get_router_settings().catalog_refresh_seconds
    mq_client = RabbitMQClient.from_settings(rabbit)
    stop = install_stop_signals()
    set_ready(lifecycle.ready_file, False)

    print(f"[*] classifiers on {get_classifier_settings().device}")
    load_classifiers()
    try:
        # Postgres may not be up yet, or the migrations / seed may not have run yet:
        # wait for a valid catalog instead of crash-looping.
        await retry(load_router_catalog, "loading the model catalog", stop_event=stop)
        await mq_client.connect(service_name="Router", stop_event=stop)
    except Stopped:
        print("[*] stopped before start-up finished")
        return

    async def wrapper(message):
        async with message.process():
            left = remaining_seconds(message)
            if left is not None and left <= 0:
                print(f"[!] {message.correlation_id}: deadline passed, dropping (caller gave up)")
                return
            try:
                await route_message(message)
            except Exception as e:
                # Answer the Gateway right away instead of letting it wait for its timeout.
                # The message is still acked: a request that can't be routed would fail
                # the same way on every redelivery.
                await mq_client.reply_error(message, f"routing failed: {e}")

    await mq_client.consume_messages(rabbit.middleware_queue, wrapper, prefetch_count=1)
    refresher = asyncio.create_task(refresh_catalog_forever())
    set_ready(lifecycle.ready_file, True)
    print(f"[*] Router listening on {rabbit.middleware_queue} (catalog refresh every {refresh_seconds}s)")

    await stop.wait()
    print("[*] SIGTERM received, shutting down")
    set_ready(lifecycle.ready_file, False)
    refresher.cancel()
    await mq_client.drain(lifecycle.shutdown_grace_seconds)
    await mq_client.close()
    await dispose_engine()
    print("[OK] router stopped")


if __name__ == "__main__":
    asyncio.run(main())