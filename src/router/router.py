import asyncio
import json
import os
import sys

from shared.rabbitmq import RabbitMQClient
from shared.utils import model_to_queue


MIDDLEWARE_QUEUE = os.getenv("MIDDLEWARE_QUEUE", "router.requests")
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "qwen2.5:7b")

SUPPORTED_MODELS_ENV = os.getenv("SUPPORTED_MODELS")

SUPPORTED_MODELS = set(m.strip() for m in SUPPORTED_MODELS_ENV.split(","))

mq_client = RabbitMQClient(
    host=os.getenv("RABBITMQ_HOST", "localhost"),
    user=os.getenv("RABBITMQ_USER", "guest"),
    password=os.getenv("RABBITMQ_PASS", "guest"),
)


async def route_message(message):
    """
    Receive a request from the Gateway and forward it
    to the appropriate worker queue.
    """
    try:
        payload = json.loads(message.body.decode())
        requested_model = payload.get("model")

        if not requested_model:
            requested_model = DEFAULT_MODEL

        # Decide which model should handle the request.
        if requested_model in SUPPORTED_MODELS:
            selected_model = requested_model
        else:
            print(f"[!] No worker for model '{requested_model}'. Falling back to '{DEFAULT_MODEL}'.")
            selected_model = DEFAULT_MODEL

        worker_queue = model_to_queue(selected_model)

        # Make sure the worker knows which model was actually selected.
        payload["model"] = selected_model

        print(f"[*] Routing request {message.correlation_id} model={selected_model} -> {worker_queue}")

        # IMPORTANT:
        #
        # forward() preserves:
        #    correlation_id
        #    reply_to
        #
        # Therefore the Worker can reply directly to
        # the Gateway's private callback queue.
        await mq_client.forward(message=message, routing_key=worker_queue, payload=payload)

    except json.JSONDecodeError:
        print("[!] Received invalid JSON")
        raise

    except Exception as e:
        print(f"[!] Error routing message: {e}")
        raise


async def main():
    sys.stdout.reconfigure(line_buffering=True)

    await mq_client.connect(service_name="Middleware / Router")
    print(f"[*] Middleware listening on {MIDDLEWARE_QUEUE}")

    # Only process one request at a time for now.
    #
    # We can increase this later if your router performs
    # more sophisticated routing.
    await mq_client.channel.set_qos(prefetch_count=1)

    # This is the queue the Gateway sends requests to.
    #
    # durable=True means the queue survives a RabbitMQ restart.
    queue = await mq_client.channel.declare_queue(MIDDLEWARE_QUEUE, durable=True)

    async def wrapper(message):
        async with message.process():
            await route_message(message)

    await queue.consume(wrapper)

    print(f"[*] Waiting for requests on {MIDDLEWARE_QUEUE}")

    try:
        # Keep the service alive.
        await asyncio.Future()
    finally:
        await mq_client.close()


if __name__ == "__main__":
    asyncio.run(main())