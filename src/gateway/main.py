import asyncio
import sys

from fastapi import Body, FastAPI, HTTPException
from fastapi.responses import JSONResponse

from shared.config import get_gateway_settings, get_rabbitmq_settings
from shared.lifecycle import Stopped
from shared.rabbitmq import RabbitMQClient, RpcError, RpcTimeout

app = FastAPI(title="MoE-Router Gateway")

# built in startup() from shared.config, so importing this module needs no env
mq_client: RabbitMQClient | None = None
_connect_task: asyncio.Task | None = None


def _rpc_ready() -> bool:
    return mq_client is not None and mq_client.is_connected and mq_client.callback_queue is not None


async def _connect():
    try:
        await mq_client.connect(service_name="API Gateway")
        await mq_client.setup_rpc_client()
    except Stopped:
        pass


@app.on_event("startup")
async def startup():
    # Connect in the background (with backoff): the HTTP server starts right away, so
    # /health answers while RabbitMQ is still unreachable and /ready says 503 until then.
    global mq_client, _connect_task
    sys.stdout.reconfigure(line_buffering=True)
    mq_client = RabbitMQClient.from_settings(get_rabbitmq_settings())
    _connect_task = asyncio.create_task(_connect())


@app.on_event("shutdown")
async def shutdown():
    # uvicorn has already stopped accepting connections and waited for in-flight
    # requests; close() fails anything still waiting for a reply.
    if mq_client:
        await mq_client.close()
    if _connect_task and not _connect_task.done():
        _connect_task.cancel()


@app.get("/health")
async def health():
    """Liveness: the process is up."""
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    """Readiness: connected to RabbitMQ with a reply queue, so /generate can work."""
    if _rpc_ready():
        return {"status": "ready"}
    return JSONResponse({"status": "not ready", "detail": "not connected to RabbitMQ"}, status_code=503)


@app.post("/generate")
async def generate(prompt: str = Body(..., embed=True)):
    if not _rpc_ready():
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
