"""The memory service's message handling (src/memory/service.py): ack after indexing, retries,
dead-lettering. The indexer itself runs against Postgres in tests/test_database_memory.py."""
import asyncio
import json

import pytest

from memory import service
from memory.indexer import IndexResult


class Message:
    def __init__(self, body):
        self.body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.acked = self.rejected = False

    async def ack(self):
        self.acked = True

    async def reject(self, requeue=False):
        assert requeue is False
        self.rejected = True


@pytest.fixture(autouse=True)
def no_backoff(monkeypatch):
    monkeypatch.setattr(service, "RETRY_BASE_SECONDS", 0)


def run_with(monkeypatch, outcomes, message, max_attempts=3):
    """outcomes: what index_turn does on each call (an exception to raise or a result)."""
    calls = []

    async def index_turn(turn_id, embedder):
        calls.append(turn_id)
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(service, "index_turn", index_turn)
    asyncio.run(service.handle_message(message, embedder=None, max_attempts=max_attempts))
    return calls


def test_indexed_turn_is_acked(monkeypatch):
    msg = Message({"turn_id": 5})
    assert run_with(monkeypatch, [IndexResult(5, "indexed", 3)], msg) == [5]
    assert msg.acked and not msg.rejected


def test_a_transient_failure_is_retried(monkeypatch):
    msg = Message({"turn_id": 5})
    calls = run_with(monkeypatch, [ConnectionError("ollama restarting"), IndexResult(5, "indexed", 1)], msg)
    assert calls == [5, 5] and msg.acked


def test_a_turn_that_keeps_failing_is_dead_lettered(monkeypatch):
    msg = Message({"turn_id": 5})
    calls = run_with(monkeypatch, [RuntimeError("boom")] * 3, msg)
    assert len(calls) == 3 and msg.rejected and not msg.acked


@pytest.mark.parametrize("body", [b"not json", {"no_turn_id": 1}, {"turn_id": "x"}])
def test_malformed_messages_are_dead_lettered_at_once(monkeypatch, body):
    msg = Message(body)
    assert run_with(monkeypatch, [], msg) == []
    assert msg.rejected
