"""route_message (src/router/router.py) with a session: fake classifiers, session store and queue."""
import asyncio
import json
import uuid
from dataclasses import dataclass

import pytest

from router import router
from router.routing_rule import Model
from router.session_routing import SessionState
from shared.memory.assembly import TurnText
from shared.prompts import SYSTEM_PROMPT

CATALOG = [
    Model("gemma3:4b", category="open-qa", complexity_tier="high", is_default=True),
    Model("coder:1b", category="coding", complexity_tier="low", context_window=8192, max_output_tokens=1024),
    Model("rewrite:3b", category="text-rewrite", complexity_tier="high"),
]


@dataclass
class Domain:
    category: str
    confidence: float
    probabilities: dict


@dataclass
class Complexity:
    level: str
    score: float


class FakeMQ:
    def __init__(self):
        self.forwarded = []

    async def forward(self, message, routing_key, payload):
        self.forwarded.append((routing_key, payload))
        return True


class Message:
    correlation_id = "c1"

    def __init__(self, payload):
        self.body = json.dumps(payload).encode()


@pytest.fixture
def env(monkeypatch):
    mq = FakeMQ()
    monkeypatch.setattr(router, "mq_client", mq)
    monkeypatch.setattr(router, "_CATALOG", CATALOG)
    calls = []

    def classify(prompt):          # what the cluster's classifier said for "fă-o descrescătoare"
        calls.append(prompt)
        return (Domain("text-rewrite", 0.91, {"text-rewrite": 0.91, "coding": 0.01}),
                Complexity("high", 0.52))

    monkeypatch.setattr(router, "classify", classify)
    return mq, calls, monkeypatch


def route(payload):
    asyncio.run(router.route_message(Message(payload)))


def with_session(monkeypatch, state, turns):
    async def load(session_id):
        return state, turns
    monkeypatch.setattr(router, "load_session", load)


def test_follow_up_stays_on_the_session_model_and_sees_the_previous_turn(env):
    mq, calls, monkeypatch = env
    with_session(monkeypatch, SessionState("coding", 0.91, 0.17),
                 [TurnText(1, "scrie o funcție de sortare", "def sort_list(lst): ...")])
    route({"prompt": "fă-o descrescătoare", "session_id": str(uuid.uuid4())})

    ((queue, payload),) = mq.forwarded
    assert calls == []                                          # classifiers skipped
    assert queue == "worker.coder-1b"
    assert (payload["inherited"], payload["session_routing"], payload["effective_domain"]) == \
        (True, "short-prompt", "coding")
    assert payload["task_type"] is None
    assert payload["turn"] == 2 and payload["recent_turns"] == 1
    assert (payload["num_ctx"], payload["max_tokens"]) == (8192, 1024)
    assert [m["content"] for m in payload["messages"]] == [
        SYSTEM_PROMPT, "scrie o funcție de sortare", "def sort_list(lst): ...", "fă-o descrescătoare"]


def test_new_session_is_classified_and_starts_at_turn_1(env):
    mq, calls, monkeypatch = env
    with_session(monkeypatch, None, [])
    route({"prompt": "rescrie textul acesta mai formal te rog frumos", "session_id": str(uuid.uuid4())})

    ((queue, payload),) = mq.forwarded
    assert calls and queue == "worker.rewrite-3b"
    assert (payload["session_routing"], payload["inherited"], payload["turn"]) == ("first-turn", False, 1)
    assert payload["task_type"] == "text-rewrite"
    assert len(payload["messages"]) == 2


def test_without_session_id_nothing_is_loaded(env):
    mq, _, monkeypatch = env

    async def load(session_id):
        raise AssertionError("no session to load")
    monkeypatch.setattr(router, "load_session", load)
    route({"prompt": "rescrie textul acesta mai formal te rog frumos"})
    assert "turn" not in mq.forwarded[0][1]


def test_session_store_down_still_routes(env):
    mq, _, monkeypatch = env

    async def load(session_id):
        raise ConnectionError("postgres down")
    monkeypatch.setattr(router, "load_session", load)
    route({"prompt": "rescrie textul acesta mai formal te rog frumos", "session_id": str(uuid.uuid4())})

    ((_, payload),) = mq.forwarded
    assert "turn" not in payload                                # the worker numbers it
    assert len(payload["messages"]) == 2
