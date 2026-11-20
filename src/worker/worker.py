"""
worker.py -- the generic worker: one Deployment serves every enabled model of the registry.

Every WORKER_SYNC_SECONDS it re-reads the enabled models (Postgres `models`) and checks each one's
engine. For every model whose engine answers it consumes that model's queue, worker.<model>, on a
channel of its own (WORKER_PREFETCH requests per model at once); a model whose engine stops
answering, or that is disabled, stops being consumed until it is back. Its queue stays declared,
so the router's requests wait there (until their deadline) instead of being dropped.

Where a model runs comes from the registry: models.inference_host, or OLLAMA_URL when empty.
The router is unchanged: it still sends a request to the queue of the model it chose, and the
reply goes straight back to the gateway (reply_to + correlation_id).
"""
import asyncio
import sys
import time
import uuid
from dataclasses import dataclass

from database import (
    ChatSessionRepository,
    ModelRepository,
    TurnAlreadyExistsError,
    TurnRepository,
    dispose_engine,
    get_session,
)
from engine import ChatResult, InferenceEngine, get_engine
from shared.config import get_lifecycle_settings, get_rabbitmq_settings, get_worker_settings
from shared.lifecycle import Stopped, install_stop_signals, retry, set_ready
from shared.prompts import single_turn_messages
from shared.rabbitmq import RabbitMQClient
from shared.utils import model_to_queue

_CONSUMED_FIELDS = {"prompt", "messages", "model", "num_ctx", "max_tokens", "session_id", "turn"}
_ROLES = {"system", "user", "assistant"}

_TURN_CONTEXT_FIELDS = ("rag_variant", "recent_turns", "retrieved_chunk_ids")

HEALTH_TIMEOUT_SECONDS = 10

OLLAMA_URL: str = ""            # set from WorkerSettings in main()
PREFETCH: int = 1
MEMORY_QUEUE: str = ""          # "" = nothing is published for indexing
SHUTDOWN_GRACE_SECONDS: float = 25
mq_client: RabbitMQClient | None = None


@dataclass
class ServedModel:
    """One enabled model of the registry, as this pod serves it."""
    model_id: str
    engine_type: str
    host: str | None
    context_window: int          # used when a payload has no num_ctx / max_tokens
    max_output_tokens: int
    engine: InferenceEngine | None = None
    problem: str | None = None   # why it cannot be served at all (no host, unknown engine)
    healthy: bool | None = None  # None until the first check
    channel: object = None       # its own channel: its prefetch doesn't limit the other models
    consumer: tuple | None = None   # (queue, consumer tag) while consuming
    inflight: int = 0

    @property
    def queue_name(self) -> str:
        return model_to_queue(self.model_id)

    @property
    def location(self) -> tuple[str, str | None]:
        """What the engine is built from: a change means a new engine and a fresh consumer."""
        return self.engine_type, self.host


MODELS: dict[str, ServedModel] = {}
_background: set[asyncio.Task] = set()


def served_from_row(row: dict, ollama_url: str) -> ServedModel:
    """A registry row (ModelRegistryEntry.to_dict()) -> the model with its engine."""
    host = row.get("inference_host") or (ollama_url if row["engine"] == "ollama" else None)
    model = ServedModel(model_id=row["model_id"], engine_type=row["engine"], host=host,
                        context_window=row["context_window"], max_output_tokens=row["max_output_tokens"])
    if host is None:
        model.problem = f"engine {row['engine']!r} needs an inference_host in the registry"
        return model
    try:
        model.engine = get_engine(row["engine"], host)
    except ValueError as e:
        model.problem = str(e)
    return model


# ------------------------------------------------------------------ one request
def messages_from_payload(payload: dict) -> list[dict]:
    """The messages for the model. The router sends `messages` (the assembled context);
    a payload with only a `prompt` (an older router) becomes [system, user]."""
    messages = payload.get("messages")
    if messages is None:
        prompt = payload.get("prompt", "")
        if not prompt:
            raise ValueError("Payload needs 'messages' or a non-empty 'prompt'.")
        return single_turn_messages(prompt)

    if not isinstance(messages, list) or not messages:
        raise ValueError("'messages' must be a non-empty list.")
    for m in messages:
        if not isinstance(m, dict) or m.get("role") not in _ROLES or not isinstance(m.get("content"), str):
            raise ValueError(f"invalid message {m!r}: need a role in {sorted(_ROLES)} and a string content")
    if messages[-1]["role"] != "user":
        raise ValueError("the last message must be the user's prompt")
    return messages


def limits_from_payload(payload: dict, model: ServedModel) -> tuple[int, int]:
    """(num_ctx, max_tokens): the ones the router sent, else the model's own from the registry."""
    num_ctx = payload.get("num_ctx") or model.context_window
    max_tokens = payload.get("max_tokens") or model.max_output_tokens
    if not (isinstance(num_ctx, int) and isinstance(max_tokens, int) and 0 < max_tokens < num_ctx):
        raise ValueError(f"need 0 < max_tokens < num_ctx, got max_tokens={max_tokens!r}, num_ctx={num_ctx!r}")
    return num_ctx, max_tokens


async def store_turn(payload: dict, user_content: str, result: ChatResult, model_id: str) -> tuple[int, int] | None:
    """Write W1: the session (created on its first turn), the turn and the session's routing state,
    in one transaction, BEFORE the reply -- the user's next prompt always sees this turn.
    Returns (turn number, row id). Without a session_id (an older client) nothing is stored."""
    if not payload.get("session_id"):
        return None
    session_id = uuid.UUID(str(payload["session_id"]))

    predicted_domain = payload.get("task_type")
    predicted_complexity = payload.get("complexity_score")
    effective_domain = payload.get("effective_domain", predicted_domain)
    effective_complexity = payload.get("effective_complexity", predicted_complexity)

    async with get_session() as session:
        turns = TurnRepository(session)
        sessions = ChatSessionRepository(session)
        await sessions.ensure(session_id)
        turn = payload.get("turn") or await turns.last_turn_number(session_id) + 1
        turn_id = await turns.create(
            session_id, turn, user_content, result.text,
            model_id=model_id,
            predicted_domain=predicted_domain,
            predicted_complexity=predicted_complexity,
            effective_domain=effective_domain,
            effective_complexity=effective_complexity,
            inherited=bool(payload.get("inherited", False)),
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            **{k: payload[k] for k in _TURN_CONTEXT_FIELDS if k in payload},
        )
        await sessions.update_routing_state(
            session_id, effective_domain,
            payload.get("effective_domain_confidence", payload.get("task_type_confidence")),
            effective_complexity,
        )
    return turn, turn_id


async def publish_for_indexing(turn_id: int) -> None:
    """The stored turn goes to the memory service (chunking + embeddings), asynchronously. If the
    publish fails, the turn stays in `turns` without chunks and `memory.service --reindex` picks it up."""
    if not MEMORY_QUEUE or mq_client is None:
        return
    try:
        await mq_client.publish({"turn_id": turn_id}, routing_key=MEMORY_QUEUE, persistent=True)
    except Exception as e:
        print(f" [!] turn {turn_id} not sent for indexing: {type(e).__name__}: {e}")


async def handle_request(payload: dict, model: ServedModel) -> dict:
    """A request from worker.<model>: the queue decides the model."""
    messages = messages_from_payload(payload)
    num_ctx, max_tokens = limits_from_payload(payload, model)

    # Another model in the payload than the queue's is a routing bug: logged, the queue wins.
    requested_model = payload.get("model")
    if requested_model and requested_model != model.model_id:
        print(f" [!] Payload asks for '{requested_model}' but came on {model.queue_name} "
              f"-- using '{model.model_id}'.")

    routing = {k: v for k, v in payload.items() if k not in _CONSUMED_FIELDS}

    prompt = messages[-1]["content"]
    print(f" [x] {model.model_id} via {model.engine_type} | routing={routing} | {len(messages)} messages, "
          f"num_ctx={num_ctx} | prompt={prompt[:50]!r}")

    started = time.perf_counter()
    result = await model.engine.chat(model=model.model_id, messages=messages, max_tokens=max_tokens,
                                     num_ctx=num_ctx)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    print(f" [OK] {model.model_id} answered in {latency_ms} ms ({len(result.text)} chars, "
          f"{result.prompt_tokens} prompt / {result.completion_tokens} completion tokens"
          f"{', truncated' if result.truncated else ''})")

    # The turn is stored with the user's original prompt, not the assembled context. If storing
    # fails the user still gets the answer; the turn is only missing from the history.
    turn = None
    try:
        stored = await store_turn(payload, payload.get("prompt") or prompt, result, model.model_id)
        if stored:
            turn, turn_id = stored
            await publish_for_indexing(turn_id)
    except TurnAlreadyExistsError as e:
        print(f" [!] turn not stored: {e} (two prompts of the session sent in parallel)")
    except Exception as e:
        print(f" [!] turn not stored: {type(e).__name__}: {e}")

    return {
        "response": result.text,
        "model": model.model_id,
        "engine": model.engine_type,
        "latency_ms": latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "truncated": result.truncated,
        "turn": turn,
        "routing": routing,
    }


# ------------------------------------------------------------------ the models served
async def load_enabled_models() -> list[dict]:
    async with get_session() as session:
        return [entry.to_dict() for entry in await ModelRepository(session).list_enabled()]


async def check_health(model: ServedModel) -> str | None:
    """None when the model can answer, else the reason."""
    if model.engine is None:
        return model.problem
    try:
        await asyncio.wait_for(model.engine.health(model.model_id), timeout=HEALTH_TIMEOUT_SECONDS)
        return None
    except Exception as e:
        return f"{type(e).__name__}: {e}" if str(e) else type(e).__name__


async def start_consuming(model: ServedModel) -> None:
    async def handle(payload: dict) -> dict:
        model.inflight += 1
        try:
            return await handle_request(payload, model)
        finally:
            model.inflight -= 1

    if model.channel is None:
        model.channel = await mq_client.open_channel()
    model.consumer = await mq_client.consume_rpc(model.queue_name, handle, prefetch_count=PREFETCH,
                                                 channel=model.channel)


async def stop_consuming(model: ServedModel) -> None:
    """No new requests; the ones being handled finish and are answered."""
    if model.consumer is not None:
        await mq_client.cancel_consumer(*model.consumer)
        model.consumer = None


async def retire(model: ServedModel) -> None:
    """The model is no longer served by this pod: stop consuming, close its channel once idle."""
    await stop_consuming(model)
    if model.channel is None:
        return

    async def close_when_idle():
        deadline = time.monotonic() + SHUTDOWN_GRACE_SECONDS
        while model.inflight and time.monotonic() < deadline:
            await asyncio.sleep(1)
        try:
            await model.channel.close()
        except Exception as e:
            print(f" [!] {model.model_id}: could not close its channel: {e}")

    task = asyncio.create_task(close_when_idle())
    _background.add(task)
    task.add_done_callback(_background.discard)


async def apply_health(model: ServedModel, problem: str | None) -> None:
    """Consume while the engine answers; log only the changes."""
    if problem is None and model.consumer is None:
        await start_consuming(model)
        print(f" [OK] {model.model_id}: serving {model.queue_name} ({model.engine_type} at {model.host})")
    elif problem is not None and model.consumer is not None:
        await stop_consuming(model)
        print(f" [!] {model.model_id}: unavailable, stopped consuming {model.queue_name}: {problem}")
    elif problem is not None and model.healthy is None:
        print(f" [!] {model.model_id}: not served yet: {problem}")
    model.healthy = problem is None


async def sync_models() -> None:
    """One pass: follow the registry's enabled models and consume the queues of those that answer."""
    wanted = {row["model_id"]: row for row in await load_enabled_models()}

    for model_id in [m for m in MODELS if m not in wanted]:
        print(f" [*] {model_id}: no longer enabled, stopped consuming")
        await retire(MODELS.pop(model_id))

    for model_id, row in wanted.items():
        fresh = served_from_row(row, OLLAMA_URL)
        current = MODELS.get(model_id)
        if current is not None and current.location != fresh.location:
            print(f" [*] {model_id}: moved to {fresh.engine_type} at {fresh.host}")
            await retire(MODELS.pop(model_id))
            current = None
        if current is None:
            MODELS[model_id] = fresh
            # requests routed to it before it answers wait in its queue instead of being dropped
            await mq_client.declare_queue(fresh.queue_name)
        else:
            current.context_window, current.max_output_tokens = fresh.context_window, fresh.max_output_tokens

    models = list(MODELS.values())
    problems = await asyncio.gather(*(check_health(m) for m in models))
    for model, problem in zip(models, problems):
        await apply_health(model, problem)


async def sync_forever(interval: float) -> None:
    while True:
        await asyncio.sleep(interval)
        try:
            await sync_models()
        except Exception as e:      # e.g. Postgres briefly down: keep serving what we have
            print(f" [!] model sync failed, keeping the current models: {type(e).__name__}: {e}")


async def main():
    global OLLAMA_URL, PREFETCH, MEMORY_QUEUE, SHUTDOWN_GRACE_SECONDS, mq_client
    sys.stdout.reconfigure(line_buffering=True)
    cfg = get_worker_settings()
    rabbit = get_rabbitmq_settings()
    lifecycle = get_lifecycle_settings()
    OLLAMA_URL, PREFETCH, MEMORY_QUEUE = cfg.ollama_url, cfg.prefetch, rabbit.memory_queue
    SHUTDOWN_GRACE_SECONDS = lifecycle.shutdown_grace_seconds
    mq_client = RabbitMQClient.from_settings(rabbit)
    stop = install_stop_signals()
    set_ready(lifecycle.ready_file, False)

    try:
        await mq_client.connect(service_name="Worker", stop_event=stop)
        if MEMORY_QUEUE:
            # declared here too, so turns published before the memory service first starts are kept
            await mq_client.declare_with_dead_letters(MEMORY_QUEUE)
        await retry(sync_models, "reading the enabled models", stop_event=stop)
    except Stopped:
        print(" [*] stopped before start-up finished")
        return

    set_ready(lifecycle.ready_file, True)
    syncer = asyncio.create_task(sync_forever(cfg.sync_seconds))
    serving = sum(m.consumer is not None for m in MODELS.values())
    print(f" [*] Worker serving {serving}/{len(MODELS)} enabled model(s) (prefetch {PREFETCH} per model, "
          f"registry re-read every {cfg.sync_seconds:g}s, default Ollama {OLLAMA_URL}).")

    await stop.wait()
    print(" [*] SIGTERM received, shutting down")
    set_ready(lifecycle.ready_file, False)
    syncer.cancel()
    await mq_client.drain(lifecycle.shutdown_grace_seconds)
    await mq_client.close()
    await dispose_engine()
    print(" [OK] worker stopped")


if __name__ == "__main__":
    asyncio.run(main())
