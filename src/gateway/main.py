import sys
import os
from fastapi import FastAPI, HTTPException, Body
from shared.rabbitmq import RabbitMQClient

app = FastAPI(title="MoE-Router Gateway")

mq_client = RabbitMQClient(
    host=os.getenv("RABBITMQ_HOST", "localhost"),
    user=os.getenv("RABBITMQ_USER", "guest"),
    password=os.getenv("RABBITMQ_PASS", "guest")
)

@app.on_event("startup")
async def startup():
    sys.stdout.reconfigure(line_buffering=True)
    await mq_client.connect(service_name="API Gateway")
    await mq_client.setup_rpc_client()

@app.on_event("shutdown")
async def shutdown():
    await mq_client.close()

@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/generate")
async def generate(prompt: str = Body(..., embed=True), model: str = Body("qwen2.5:7b", embed=True)):
    try:
        routing_key = f"queue_{model.replace(':', '_').replace('.', '_').replace('/', '_')}"
        return await mq_client.call({"prompt": prompt, "model": model}, routing_key=routing_key)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))