"""The gateway's /generate (src/gateway/main.py): session ids, with a fake RabbitMQ client."""
import uuid

import pytest
from fastapi.testclient import TestClient

from gateway import main


class FakeMQ:
    is_connected = True
    callback_queue = object()

    def __init__(self):
        self.calls = []

    async def call(self, payload, routing_key, timeout):
        self.calls.append(payload)
        return {"response": "ok", "model": "gemma3:1b"}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setenv("RABBITMQ_USER", "u")
    monkeypatch.setenv("RABBITMQ_PASS", "p")
    main.get_rabbitmq_settings.cache_clear()
    mq = FakeMQ()
    monkeypatch.setattr(main, "mq_client", mq)
    yield TestClient(main.app), mq        # no `with`: the startup hook (real RabbitMQ) doesn't run
    main.get_rabbitmq_settings.cache_clear()


def test_new_conversation_gets_a_session_id(client):
    http, mq = client
    body = http.post("/generate", json={"prompt": "salut"}).json()

    session_id = uuid.UUID(body["session_id"])          # a valid uuid
    assert mq.calls == [{"prompt": "salut", "session_id": str(session_id)}]
    assert body["response"] == "ok"


def test_given_session_id_is_passed_through(client):
    http, mq = client
    sid = str(uuid.uuid4())
    body = http.post("/generate", json={"prompt": "continua", "session_id": sid}).json()
    assert body["session_id"] == sid
    assert mq.calls[0]["session_id"] == sid


@pytest.mark.parametrize("payload", [
    {"prompt": "x", "session_id": "not-a-uuid"},
    {"prompt": ""},
    {"session_id": str(uuid.uuid4())},
])
def test_invalid_requests_are_rejected(client, payload):
    http, mq = client
    assert http.post("/generate", json=payload).status_code == 422
    assert mq.calls == []
