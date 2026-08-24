import sys
import os
import asyncio
import httpx
from shared.rabbitmq import RabbitMQClient
from shared.utils import model_to_queue

MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b")
QUEUE_NAME = model_to_queue(MODEL_NAME)
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

mq_client = RabbitMQClient(
    host=os.getenv("RABBITMQ_HOST", "localhost"),
    user=os.getenv("RABBITMQ_USER", "guest"),
    password=os.getenv("RABBITMQ_PASS", "guest")
)


async def handle_request(payload: dict) -> dict:
    """Business logic for the worker."""
    prompt = payload.get("prompt", "")
    requested_model = payload.get("model", MODEL_NAME)
    print(f" [x] Processing: {prompt[:50]}...")

    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={"model": requested_model, "prompt": prompt, "stream": False},
        )
        response.raise_for_status()
        result = response.json()

    return {
        "response": result.get("response", ""),
        "model": requested_model,
    }


async def main():
    sys.stdout.reconfigure(line_buffering=True)
    await mq_client.connect(service_name=f"Worker ({MODEL_NAME})")

    print(f" [*] Worker listening on {QUEUE_NAME}. To exit press CTRL+C")
    await mq_client.consume_rpc(queue_name=QUEUE_NAME, handler_func=handle_request)

    try:
        await asyncio.Future()
    finally:
        await mq_client.close()


if __name__ == "__main__":
    asyncio.run(main())