import sys
import os
import json
import asyncio
import aio_pika
import httpx

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "localhost")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "guest")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "guest")

REQUEST_QUEUE = "task_queue"
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")


async def call_ollama(prompt: str):
    async with httpx.AsyncClient(timeout=60.0) as client:
        response = await client.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": "qwen2.5:7b",
                "prompt": prompt,
                "stream": False,
            },
        )
        response.raise_for_status()
        return response.json()


async def process_message(message: aio_pika.IncomingMessage, channel):
    async with message.process():
        try:
            body = json.loads(message.body.decode())
            prompt = body.get("prompt", "")

            print(f" [x] Processing prompt: {prompt[:50]}...")

            result = await call_ollama(prompt)

            response_payload = {
                "task_id": message.correlation_id,
                "response": result.get("response", ""),
                "model": result.get("model", "qwen2.5:7b"),
            }

            if message.reply_to:
                await channel.default_exchange.publish(
                    aio_pika.Message(
                        body=json.dumps(response_payload).encode(),
                        correlation_id=message.correlation_id,
                    ),
                    routing_key=message.reply_to,
                )

            print(" [x] Done.")

        except Exception as e:
            print(f" [!] Error processing message: {e}")

            if message.reply_to:
                try:
                    error_response = {
                        "task_id": message.correlation_id,
                        "error": str(e),
                    }

                    await channel.default_exchange.publish(
                        aio_pika.Message(
                            body=json.dumps(error_response).encode(),
                            correlation_id=message.correlation_id,
                        ),
                        routing_key=message.reply_to,
                    )

                except Exception as ex:
                    print(f" [!] Failed to send error reply: {ex}")


async def main():
    sys.stdout.reconfigure(line_buffering=True)

    connection = None
    max_retries = 10

    for attempt in range(max_retries):
        try:
            print(
                f"[*] Încercare conectare RabbitMQ pe "
                f"{RABBITMQ_HOST} ({attempt + 1}/{max_retries})..."
            )

            connection = await aio_pika.connect_robust(
                host=RABBITMQ_HOST,
                login=RABBITMQ_USER,
                password=RABBITMQ_PASS,
            )

            print("[*] Conexiune RabbitMQ stabilită cu succes!")
            break

        except Exception as e:
            print(f"[!] Conexiune eșuată: {e}. Aștept 5 secunde...")
            await asyncio.sleep(5)

    if not connection:
        print("[!] Eșec critic: Nu s-a putut conecta la RabbitMQ.")
        sys.exit(1)

    channel = await connection.channel()
    await channel.set_qos(prefetch_count=1)

    queue = await channel.declare_queue(
        REQUEST_QUEUE,
        durable=True,
    )

    print(" [*] Worker waiting for messages. To exit press CTRL+C")

    async def on_message(message: aio_pika.IncomingMessage):
        await process_message(message, channel)

    await queue.consume(on_message)

    try:
        await asyncio.Future()
    finally:
        await connection.close()


if __name__ == "__main__":
    asyncio.run(main())