import sys
import os
import json
import uuid
import asyncio

from fastapi import FastAPI, HTTPException, Body
import aio_pika


app = FastAPI(title="MoE-Router Gateway")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")

REQUEST_QUEUE_DEFAULT = "task_queue"


class RabbitMQClient:
    def __init__(self):
        self.connection = None
        self.channel = None
        self.callback_queue = None
        self.exchange = None
        self.futures = {}

    async def connect(self):
        max_retries = 10

        for attempt in range(max_retries):
            try:
                print(
                    f"[*] API Gateway - Încercare conectare RabbitMQ "
                    f"({attempt + 1}/{max_retries})..."
                )

                self.connection = await aio_pika.connect_robust(
                    host=RABBITMQ_HOST,
                    login=RABBITMQ_USER,
                    password=RABBITMQ_PASS,
                )

                print("[*] API Gateway - Conexiune RabbitMQ stabilită cu succes!")
                break

            except Exception as e:
                print(
                    f"[!] Conexiune eșuată: {e}. "
                    f"Reîncerc în 5 secunde..."
                )
                await asyncio.sleep(5)

        if not self.connection:
            raise Exception(
                "Nu m-am putut conecta la RabbitMQ după mai multe încercări."
            )

        # Canal
        self.channel = await self.connection.channel()
        self.exchange = self.channel.default_exchange

        self.callback_queue = await self.channel.declare_queue(
            exclusive=True,
            auto_delete=True,
        )

        await self.callback_queue.consume(
            self.on_response,
            no_ack=True,
        )

        print(
            f"[*] Callback queue: {self.callback_queue.name}"
        )

        return self

    async def on_response(self, message: aio_pika.IncomingMessage):
        correlation_id = message.correlation_id

        if not correlation_id:
            print("[!] Am primit un mesaj fără correlation_id")
            return

        future = self.futures.pop(correlation_id, None)

        if future is None:
            print(
                f"[!] Nu există future pentru correlation_id={correlation_id}"
            )
            return

        try:
            response = json.loads(message.body.decode())
            future.set_result(response)

        except Exception as e:
            future.set_exception(e)

    async def call(self, payload: dict, routing_key: str):
        if not self.channel:
            raise RuntimeError("RabbitMQ channel nu este conectat.")

        if not self.exchange:
            raise RuntimeError("RabbitMQ exchange nu este configurat.")

        correlation_id = str(uuid.uuid4())

        loop = asyncio.get_running_loop()
        future = loop.create_future()

        self.futures[correlation_id] = future

        try:
            await self.exchange.publish(
                aio_pika.Message(
                    body=json.dumps(payload).encode(),
                    correlation_id=correlation_id,
                    reply_to=self.callback_queue.name,
                    content_type="application/json",
                ),
                routing_key=routing_key,
            )

            print(
                f"[*] Request trimis către {routing_key}, "
                f"correlation_id={correlation_id}"
            )

            return await future

        except Exception:
            self.futures.pop(correlation_id, None)
            raise


rabbitmq_client = RabbitMQClient()


@app.on_event("startup")
async def startup():
    sys.stdout.reconfigure(line_buffering=True)
    await rabbitmq_client.connect()


@app.on_event("shutdown")
async def shutdown():
    if rabbitmq_client.connection:
        await rabbitmq_client.connection.close()


@app.post("/generate")
async def generate(
    prompt: str = Body(..., embed=True),
    model: str = Body("qwen2.5:7b", embed=True)
):
    """Forward request to worker and wait for model response."""
    try:
        # Determine routing key based on model name
        safe_model_name = model.replace(':', '_').replace('.', '_').replace('/', '_')
        routing_key = f"queue_{safe_model_name}"

        response = await rabbitmq_client.call(
            {
                "prompt": prompt,
                "model": model
            },
            routing_key=routing_key
        )
        return response

    except Exception as e:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Error processing request: {str(e)}",
        )

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "rabbitmq": bool(rabbitmq_client.connection),
    }