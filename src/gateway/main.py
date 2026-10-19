import sys

from fastapi import Body, FastAPI, HTTPException

from shared.config import get_gateway_settings, get_rabbitmq_settings
from shared.rabbitmq import RabbitMQClient, RpcError, RpcTimeout

app = FastAPI(title="MoE-Router Gateway")

# built in startup() from shared.config, so importing this module needs no env
mq_client: RabbitMQClient | None = None


@app.on_event("startup")
async def startup():
    global mq_client
    sys.stdout.reconfigure(line_buffering=True)
    mq_client = RabbitMQClient.from_settings(get_rabbitmq_settings())
    await mq_client.connect(service_name="API Gateway")
    await mq_client.setup_rpc_client()


@app.on_event("shutdown")
async def shutdown():
    if mq_client:
        await mq_client.close()


@app.get("/health")
async def health():
    return {"status": "ok"}

@app.post("/generate")
async def generate(prompt: str = Body(..., embed=True)):
    if mq_client is None or mq_client.channel is None or mq_client.channel.is_closed:
        raise HTTPException(status_code=503, detail="not connected to RabbitMQ")
    try:
        return await mq_client.call(
            {"prompt": prompt},
            routing_key=get_rabbitmq_settings().middleware_queue,
            timeout=get_gateway_settings().rpc_timeout_seconds,
        )
    except RpcTimeout as e:
        raise HTTPException(status_code=504, detail=str(e))
    except RpcError as e:              # the router or the worker answered with an error
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
