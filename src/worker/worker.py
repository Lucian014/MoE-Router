import asyncio
import sys
import time

from engine import get_engine
from shared.config import get_rabbitmq_settings, get_worker_settings
from shared.rabbitmq import RabbitMQClient
from shared.utils import model_to_queue

# Campurile pe care workerul le consuma; restul payload-ului e metadata de
# rutare pusa de router (domeniu, nivel, probabilitati, confidence...) si se
# intoarce neatinsa la Gateway, sub "routing". Asa workerul nu depinde de
# numele exacte ale campurilor din router.
_CONSUMED_FIELDS = {"prompt", "model"}

# Construite in main() din shared.config. Fara MODEL_NAME, WorkerSettings
# arunca o eroare de validare la pornire (build_manifests.py il seteaza mereu;
# lipsa lui e o eroare de deployment).
MODEL_NAME: str = ""
ENGINE_TYPE: str = ""
engine = None
mq_client: RabbitMQClient | None = None


async def handle_request(payload: dict) -> dict:
    prompt = payload.get("prompt", "")
    if not prompt:
        raise ValueError("Payload is missing the required 'prompt' field.")

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

    print(f" [x] {MODEL_NAME} via {ENGINE_TYPE} | routing={routing} | prompt={prompt[:50]!r}")

    started = time.perf_counter()
    text = await engine.generate(model=MODEL_NAME, prompt=prompt)
    latency_ms = round((time.perf_counter() - started) * 1000, 1)

    print(f" [OK] {MODEL_NAME} answered in {latency_ms} ms ({len(text)} chars)")

    return {
        "response": text,
        "model": MODEL_NAME,
        "engine": ENGINE_TYPE,
        "latency_ms": latency_ms,
        "routing": routing,
    }


async def main():
    global MODEL_NAME, ENGINE_TYPE, engine, mq_client
    sys.stdout.reconfigure(line_buffering=True)
    cfg = get_worker_settings()
    MODEL_NAME, ENGINE_TYPE = cfg.model_name, cfg.engine_type
    queue_name = model_to_queue(MODEL_NAME)
    engine = get_engine(ENGINE_TYPE, cfg.inference_host)
    mq_client = RabbitMQClient.from_settings(get_rabbitmq_settings())

    await mq_client.connect(service_name=f"Worker ({MODEL_NAME})")
    print(f" [*] Worker listening on {queue_name} (model={MODEL_NAME}, engine={ENGINE_TYPE}).")
    await mq_client.consume_rpc(queue_name=queue_name, handler_func=handle_request,
                                prefetch_count=cfg.prefetch)
    try:
        await asyncio.Future()
    finally:
        await mq_client.close()


if __name__ == "__main__":
    asyncio.run(main())