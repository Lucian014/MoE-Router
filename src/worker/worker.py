import asyncio
import os
import sys
import time

from shared.rabbitmq import RabbitMQClient
from shared.utils import model_to_queue
from engine import get_engine

MODEL_NAME = os.getenv("MODEL_NAME")
ENGINE_TYPE = os.getenv("ENGINE_TYPE", "ollama")
INFERENCE_HOST = os.getenv("INFERENCE_HOST", "http://localhost:11434")

if not MODEL_NAME:
    # Fara MODEL_NAME, workerul ar asculta pe coada unui model inventat.
    # build_manifests.py il seteaza mereu; lipsa lui e o eroare de deployment.
    raise SystemExit("[!] MODEL_NAME is not set -- refusing to start.")

QUEUE_NAME = model_to_queue(MODEL_NAME)

# Campurile pe care workerul le consuma; restul payload-ului e metadata de
# rutare pusa de router (domeniu, nivel, probabilitati, confidence...) si se
# intoarce neatinsa la Gateway, sub "routing". Asa workerul nu depinde de
# numele exacte ale campurilor din router.
_CONSUMED_FIELDS = {"prompt", "model"}

engine = get_engine(ENGINE_TYPE, INFERENCE_HOST)

mq_client = RabbitMQClient(
    host=os.getenv("RABBITMQ_HOST", "localhost"),
    user=os.getenv("RABBITMQ_USER", "guest"),
    password=os.getenv("RABBITMQ_PASS", "guest"),
)


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
    sys.stdout.reconfigure(line_buffering=True)
    await mq_client.connect(service_name=f"Worker ({MODEL_NAME})")
    print(f" [*] Worker listening on {QUEUE_NAME} (model={MODEL_NAME}, engine={ENGINE_TYPE}).")
    await mq_client.consume_rpc(queue_name=QUEUE_NAME, handler_func=handle_request)
    try:
        await asyncio.Future()
    finally:
        await mq_client.close()


if __name__ == "__main__":
    asyncio.run(main())