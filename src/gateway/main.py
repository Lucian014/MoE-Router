import asyncio
import sys
import uuid
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from shared.config import get_gateway_settings, get_rabbitmq_settings
from shared.lifecycle import Stopped
from shared.rabbitmq import RabbitMQClient, RpcError, RpcTimeout

app = FastAPI(title="MoE-Router Gateway")
api = APIRouter(prefix="/api")

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
    global mq_client, _connect_task
    sys.stdout.reconfigure(line_buffering=True)
    mq_client = RabbitMQClient.from_settings(get_rabbitmq_settings())
    _connect_task = asyncio.create_task(_connect())


@app.on_event("shutdown")
async def shutdown():
    if mq_client:
        await mq_client.close()
    if _connect_task and not _connect_task.done():
        _connect_task.cancel()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/ready")
async def ready():
    if _rpc_ready():
        return {"status": "ready"}
    return JSONResponse({"status": "not ready", "detail": "not connected to RabbitMQ"}, status_code=503)


class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    session_id: uuid.UUID | None = None
    variant: str | None = Field(None, pattern=r"^V[0-9]$")


@api.post("/generate")
async def generate(req: GenerateRequest):
    if not _rpc_ready():
        raise HTTPException(status_code=503, detail="not connected to RabbitMQ")
    session_id = str(req.session_id or uuid.uuid4())
    try:
        request = {"prompt": req.prompt, "session_id": session_id}
        if req.variant:
            request["variant"] = req.variant
        reply = await mq_client.call(
            request,
            routing_key=get_rabbitmq_settings().middleware_queue,
            timeout=get_gateway_settings().rpc_timeout_seconds,
        )
        return {**reply, "session_id": session_id}
    except RpcTimeout as e:
        raise HTTPException(status_code=504, detail=str(e))
    except RpcError as e:
        raise HTTPException(status_code=502, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class UIFiles(StaticFiles):

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        immutable = path.startswith("assets/") and response.status_code == 200
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable" if immutable else "no-cache"
        return response


def mount_ui(directory: Path) -> bool:
    if not directory.is_dir():
        return False
    app.mount("/", UIFiles(directory=directory, html=True), name="ui")
    return True


app.include_router(api)
app.add_api_route("/generate", generate, methods=["POST"], deprecated=True)
mount_ui(Path(__file__).with_name("static"))            # last
