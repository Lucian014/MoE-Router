import sys
import os
import asyncio
from shared.rabbitmq import RabbitMQClient
from shared.utils import model_to_queue
from engine import get_engine

MODEL_NAME = os.getenv("MODEL_NAME", "qwen2.5:7b")
ENGINE_TYPE = os.getenv("ENGINE_TYPE", "ollama")
INFERENCE_HOST = os.getenv("INFERENCE_HOST", "http://localhost:11434")
QUEUE_NAME = model_to_queue(MODEL_NAME)

engine = get_engine(ENGINE_TYPE, INFERENCE_HOST)

mq_client = RabbitMQClient(
    host=os.getenv("RABBITMQ_HOST", "localhost"),
    user=os.getenv("RABBITMQ_USER", "guest"),
    password=os.getenv("RABBITMQ_PASS", "guest")
)


async def handle_request(payload: dict) -> dict:

    prompt = payload.get("prompt", "")
    requested_model = payload.get("model", MODEL_NAME)

    print(f" [x] Processing via {ENGINE_TYPE}: {prompt[:50]}...")

    text = await engine.generate(model=requested_model, prompt=prompt)

    return {
        "response": text,
        "model": requested_model,
        "routing_score": payload.get("routing_score"),
    }


async def main():
    sys.stdout.reconfigure(line_buffering=True)
    await mq_client.connect(service_name=f"Worker ({MODEL_NAME})")
    print(f" [*] Worker listening on {QUEUE_NAME} (engine={ENGINE_TYPE}). To exit press CTRL+C")
    await mq_client.consume_rpc(queue_name=QUEUE_NAME, handler_func=handle_request)
    try:
        await asyncio.Future()
    finally:
        await mq_client.close()


if __name__ == "__main__":
    asyncio.run(main())