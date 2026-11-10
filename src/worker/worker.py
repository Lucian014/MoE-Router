import asyncio
import sys
import time
import uuid

from database import ChatSessionRepository, TurnAlreadyExistsError, TurnRepository, dispose_engine, get_session
from engine import ChatResult, get_engine
from shared.config import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_OUTPUT_TOKENS,
    get_lifecycle_settings,
    get_rabbitmq_settings,
    get_worker_settings,
)
from shared.lifecycle import Stopped, install_stop_signals, retry, set_ready
from shared.prompts import single_turn_messages
from shared.rabbitmq import RabbitMQClient
from shared.utils import model_to_queue

_CONSUMED_FIELDS = {"prompt", "messages", "model", "num_ctx", "max_tokens", "session_id", "turn"}
_ROLES = {"system", "user", "assistant"}

_TURN_CONTEXT_FIELDS = ("rag_variant", "recent_turns", "retrieved_chunk_ids")

ENGINE_CHECK_SECONDS = 15

MODEL_NAME: str = ""
ENGINE_TYPE: str = ""
CONTEXT_WINDOW: int = DEFAULT_CONTEXT_WINDOW          # set from WorkerSettings in main()
MAX_OUTPUT_TOKENS: int = DEFAULT_MAX_OUTPUT_TOKENS
MEMORY_QUEUE: str = ""          # "" = nothing is published for indexing
engine = None
mq_client: RabbitMQClient | None = None


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


def limits_from_payload(payload: dict) -> tuple[int, int]:
    """(num_ctx, max_tokens): the chosen model's, sent by the router; else the settings'."""
    num_ctx = payload.get("num_ctx") or CONTEXT_WINDOW
    max_tokens = payload.get("max_tokens") or MAX_OUTPUT_TOKENS
    if not (isinstance(num_ctx, int) and isinstance(max_tokens, int) and 0 < max_tokens < num_ctx):
        raise ValueError(f"need 0 < max_tokens < num_ctx, got max_tokens={max_tokens!r}, num_ctx={num_ctx!r}")
    return num_ctx, max_tokens


async def store_turn(payload: dict, user_content: str, result: ChatResult) -> tuple[int, int] | None:
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
            model_id=MODEL_NAME,
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


async def handle_request(payload: dict) -> dict:
    messages = messages_from_payload(payload)
    num_ctx, max_tokens = limits_from_payload(payload)

    # One worker = one model: the worker.<model> queue belongs to this model, so we always
    # generate with MODEL_NAME. Another model in the payload is a routing bug -- logged, but
    # Ollama is not made to load another model on this worker.
    requested_model = payload.get("model")
    if requested_model and requested_model != MODEL_NAME:
        print(
            f" [!] Payload asks for '{requested_model}' but this worker serves "
            f"'{MODEL_NAME}' -- using '{MODEL_NAME}'."
        )

    routing = {k: v for k, v in payload.items() if k not in _CONSUMED_FIELDS}

    prompt = messages[-1]["content"]
    print(f" [x] {MODEL_NAME} via {ENGINE_TYPE} | routing={routing} | {len(messages)} messages, "
          f"num_ctx={num_ctx} | prompt={prompt[:50]!r}")

    started = time.perf_counter()
    result = await engine.chat(model=MODEL_NAME, messages=messages, max_tokens=max_tokens, num_ctx=num_ctx)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    print(f" [OK] {MODEL_NAME} answered in {latency_ms} ms ({len(result.text)} chars, "
          f"{result.prompt_tokens} prompt / {result.completion_tokens} completion tokens"
          f"{', truncated' if result.truncated else ''})")

    # The turn is stored with the user's original prompt, not the assembled context. If storing
    # fails the user still gets the answer; the turn is only missing from the history.
    turn = None
    try:
        stored = await store_turn(payload, payload.get("prompt") or prompt, result)
        if stored:
            turn, turn_id = stored
            await publish_for_indexing(turn_id)
    except TurnAlreadyExistsError as e:
        print(f" [!] turn not stored: {e} (two prompts of the session sent in parallel)")
    except Exception as e:
        print(f" [!] turn not stored: {type(e).__name__}: {e}")

    return {
        "response": result.text,
        "model": MODEL_NAME,
        "engine": ENGINE_TYPE,
        "latency_ms": latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "truncated": result.truncated,
        "turn": turn,
        "routing": routing,
    }


async def engine_health() -> None:
    await engine.health(MODEL_NAME)


async def watch_engine(ready_file: str) -> None:
    """Keep the readiness file in sync with the engine (e.g. Ollama restarted, model removed)."""
    healthy = True
    while True:
        await asyncio.sleep(ENGINE_CHECK_SECONDS)
        try:
            await engine_health()
            ok = True
        except Exception as e:
            ok = False
            if healthy:
                print(f" [!] engine unhealthy, marking pod not ready: {e}")
        if ok and not healthy:
            print(" [OK] engine healthy again, marking pod ready")
        healthy = ok
        set_ready(ready_file, ok)


async def main():
    global MODEL_NAME, ENGINE_TYPE, CONTEXT_WINDOW, MAX_OUTPUT_TOKENS, MEMORY_QUEUE, engine, mq_client
    sys.stdout.reconfigure(line_buffering=True)
    cfg = get_worker_settings()
    rabbit = get_rabbitmq_settings()
    lifecycle = get_lifecycle_settings()
    MODEL_NAME, ENGINE_TYPE = cfg.model_name, cfg.engine_type
    CONTEXT_WINDOW, MAX_OUTPUT_TOKENS = cfg.context_window, cfg.max_output_tokens
    MEMORY_QUEUE = rabbit.memory_queue
    queue_name = model_to_queue(MODEL_NAME)
    engine = get_engine(ENGINE_TYPE, cfg.inference_host)
    mq_client = RabbitMQClient.from_settings(rabbit)
    stop = install_stop_signals()
    set_ready(lifecycle.ready_file, False)

    try:
        # Don't take requests before the model can answer them (Ollama starting, model not pulled yet)
        await retry(engine_health, f"{ENGINE_TYPE} health check for {MODEL_NAME}", stop_event=stop)
        await mq_client.connect(service_name=f"Worker ({MODEL_NAME})", stop_event=stop)
    except Stopped:
        print(" [*] stopped before start-up finished")
        return
    if MEMORY_QUEUE:
        # declared here too, so turns published before the memory service first starts are kept
        await mq_client.declare_with_dead_letters(MEMORY_QUEUE)

    await mq_client.consume_rpc(queue_name=queue_name, handler_func=handle_request,
                                prefetch_count=cfg.prefetch)
    set_ready(lifecycle.ready_file, True)
    watcher = asyncio.create_task(watch_engine(lifecycle.ready_file))
    print(f" [*] Worker listening on {queue_name} (model={MODEL_NAME}, engine={ENGINE_TYPE}, "
          f"prefetch={cfg.prefetch}, context_window={CONTEXT_WINDOW}, max_output_tokens={MAX_OUTPUT_TOKENS}).")

    await stop.wait()
    print(" [*] SIGTERM received, shutting down")
    set_ready(lifecycle.ready_file, False)
    watcher.cancel()
    await mq_client.drain(lifecycle.shutdown_grace_seconds)
    await mq_client.close()
    await dispose_engine()
    print(" [OK] worker stopped")


if __name__ == "__main__":
    asyncio.run(main())