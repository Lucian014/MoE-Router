import os
import json
from fastapi import FastAPI, HTTPException
import aio_pika

app = FastAPI(title="RabbitMQ K8s Test")

RABBITMQ_HOST = os.getenv("RABBITMQ_HOST", "rabbitmq.backend-namespace.svc.cluster.local")
RABBITMQ_USER = os.getenv("RABBITMQ_USER", "admin")
RABBITMQ_PASS = os.getenv("RABBITMQ_PASS", "admin123")
QUEUE_NAME = "test_queue"


async def get_rabbitmq_channel():
    """Creează conexiunea asincronă către RabbitMQ din K8s."""
    connection = await aio_pika.connect_robust(
        host=RABBITMQ_HOST,
        login=RABBITMQ_USER,
        password=RABBITMQ_PASS,
        port=5672
    )
    channel = await connection.channel()
    # Declarăm coada (dacă nu există deja, o creează)
    await channel.declare_queue(QUEUE_NAME, durable=True)
    return connection, channel


@app.post("/push-test")
async def push_to_queue(message_text: str = "Test mesaj din FastAPI"):
    """Trimite un mesaj în coadă."""
    try:
        connection, channel = await get_rabbitmq_channel()
        async with connection:
            payload = {"status": "ok", "content": message_text}

            await channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(payload).encode(),
                    delivery_mode=aio_pika.DeliveryMode.PERSISTENT
                ),
                routing_key=QUEUE_NAME
            )
        return {"success": True, "pushed_message": payload}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare conexiune K8s RabbitMQ: {str(e)}")


@app.get("/pop-test")
async def pop_from_queue():
    """Scoate și returnează un mesaj din coadă."""
    try:
        connection, channel = await get_rabbitmq_channel()
        async with connection:
            queue = await channel.get_queue(QUEUE_NAME)

            # Încercăm să citim un mesaj fără să blocăm procesul (timeout/no-wait)
            message = await queue.get(no_ack=False, fail=False)

            if message is None:
                return {"success": False, "message": "Coada este goală!"}

            async with message.process():
                data = json.loads(message.body.decode())
                return {"success": True, "received_message": data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eroare la citirea din coadă: {str(e)}")