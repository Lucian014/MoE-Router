import asyncio
import sys
import time

from engine import get_engine
from shared.config import get_lifecycle_settings, get_rabbitmq_settings, get_worker_settings
from shared.lifecycle import Stopped, install_stop_signals, retry, set_ready
from shared.prompts import single_turn_messages
from shared.rabbitmq import RabbitMQClient
from shared.utils import model_to_queue

# Campurile pe care workerul le consuma; restul payload-ului e metadata de
# rutare pusa de router (domeniu, nivel, probabilitati, confidence...) si se
# intoarce neatinsa la Gateway, sub "routing". Asa workerul nu depinde de
# numele exacte ale campurilor din router.
_CONSUMED_FIELDS = {"prompt", "messages", "model", "num_ctx", "max_tokens"}
_ROLES = {"system", "user", "assistant"}

# how often the running worker re-checks its engine for the readiness probe
ENGINE_CHECK_SECONDS = 15

# Construite in main() din shared.config. Fara MODEL_NAME, WorkerSettings
# arunca o eroare de validare la pornire (chart-ul Helm il seteaza mereu;
# lipsa lui e o eroare de deployment).
MODEL_NAME: str = ""
ENGINE_TYPE: str = ""
CONTEXT_WINDOW: int = 4096
MAX_OUTPUT_TOKENS: int = 512
engine = None
mq_client: RabbitMQClient | None = None


def messages_from_payload(payload: dict) -> list[dict]:
    """Mesajele de trimis modelului. Routerul trimite `messages` (contextul asamblat);
    un payload doar cu `prompt` (routerul vechi) devine [system, user]."""
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
    """(num_ctx, max_tokens): ale modelului ales, trimise de router; altfel cele din setari."""
    num_ctx = payload.get("num_ctx") or CONTEXT_WINDOW
    max_tokens = payload.get("max_tokens") or MAX_OUTPUT_TOKENS
    if not (isinstance(num_ctx, int) and isinstance(max_tokens, int) and 0 < max_tokens < num_ctx):
        raise ValueError(f"need 0 < max_tokens < num_ctx, got max_tokens={max_tokens!r}, num_ctx={num_ctx!r}")
    return num_ctx, max_tokens


async def handle_request(payload: dict) -> dict:
    messages = messages_from_payload(payload)
    num_ctx, max_tokens = limits_from_payload(payload)

    # Un worker = un model: coada worker.<model> e a acestui model, deci
    # generam mereu cu MODEL_NAME. Daca routerul a scris alt model in payload,
    # e un bug de rutare -- il logam, dar nu fortam Ollama sa incarce alt
    # model pe workerul asta.
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

    return {
        "response": result.text,
        "model": MODEL_NAME,
        "engine": ENGINE_TYPE,
        "latency_ms": latency_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "truncated": result.truncated,
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
    global MODEL_NAME, ENGINE_TYPE, CONTEXT_WINDOW, MAX_OUTPUT_TOKENS, engine, mq_client
    sys.stdout.reconfigure(line_buffering=True)
    cfg = get_worker_settings()
    lifecycle = get_lifecycle_settings()
    MODEL_NAME, ENGINE_TYPE = cfg.model_name, cfg.engine_type
    CONTEXT_WINDOW, MAX_OUTPUT_TOKENS = cfg.context_window, cfg.max_output_tokens
    queue_name = model_to_queue(MODEL_NAME)
    engine = get_engine(ENGINE_TYPE, cfg.inference_host)
    mq_client = RabbitMQClient.from_settings(get_rabbitmq_settings())
    stop = install_stop_signals()
    set_ready(lifecycle.ready_file, False)

    try:
        # Don't take requests before the model can answer them (Ollama starting, model
        # still being pulled): they wait in the queue instead, up to their deadline.
        await retry(engine_health, f"{ENGINE_TYPE} health check for {MODEL_NAME}", stop_event=stop)
        await mq_client.connect(service_name=f"Worker ({MODEL_NAME})", stop_event=stop)
    except Stopped:
        print(" [*] stopped before start-up finished")
        return

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
    print(" [OK] worker stopped")


if __name__ == "__main__":
    asyncio.run(main())