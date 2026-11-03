"""route_message (src/router/router.py) with a session: fake classifiers, session store and queue."""
import asyncio
import json
import uuid
from dataclasses import dataclass

import pytest

from router import router
from router.routing_rule import Model
from router.session_routing import SessionState
from shared.memory.assembly import Fragment, TurnText
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


# ------------------------------------------------------------------ retrieval (V1 / V2)
LONG_ANSWER = "token " * 12000             # turn 1 can't be included whole on an 8k model


def long_session(monkeypatch):
    with_session(monkeypatch, SessionState("coding", 0.91, 0.17), [
        TurnText(1, "cum fac login cu JWT?", LONG_ANSWER),
        TurnText(2, "și refresh token?", "def refresh(): ..."),
        TurnText(3, "adaugă teste", "def test_refresh(): ..."),
    ])


def fake_retrieve(monkeypatch, result=None, error=None):
    calls = []

    async def retrieve(session_id, before_turn, prompt, preset):
        calls.append((before_turn, prompt, preset.name))
        if error:
            raise error
        return result
    monkeypatch.setattr(router, "retrieve", retrieve)
    return calls


def settings(monkeypatch, **env):
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    router.get_retrieval_settings.cache_clear()


@pytest.fixture(autouse=True)
def fresh_retrieval_settings():
    router.get_retrieval_settings.cache_clear()
    yield
    router.get_retrieval_settings.cache_clear()


def test_fragments_of_turns_not_included_whole_reach_the_model(env):
    mq, _, monkeypatch = env
    long_session(monkeypatch)
    pool = [Fragment(31, 3, 0, "def test_refresh(): ...", "adaugă teste"),     # turn 3 is whole
            Fragment(11, 1, 4, "create_access_token(sub, expires)", "cum fac login cu JWT?")]
    calls = fake_retrieve(monkeypatch, router.Retrieved(pool, candidates=9, timings={"embed_ms": 12.0}))
    route({"prompt": "cum era funcția care creează tokenul de acces?", "session_id": str(uuid.uuid4())})

    ((_, payload),) = mq.forwarded
    assert calls == [(4, "cum era funcția care creează tokenul de acces?", "V2")]
    assert payload["rag_variant"] == "V2"
    assert payload["retrieved_chunk_ids"] == [11]
    assert payload["recent_turns"] == 2
    assert payload["retrieval"] == {"candidates": 9, "pool": 2, "embed_ms": 12.0}
    excerpts = payload["messages"][1]
    assert excerpts["role"] == "system"
    assert "[turn 1] Question: cum fac login cu JWT?\ncreate_access_token(sub, expires)" in excerpts["content"]


def test_retrieval_failure_answers_with_recent_turns_as_v0(env):
    mq, _, monkeypatch = env
    long_session(monkeypatch)
    fake_retrieve(monkeypatch, error=ConnectionError("ollama down"))
    route({"prompt": "cum era funcția care creează tokenul de acces?", "session_id": str(uuid.uuid4())})

    ((_, payload),) = mq.forwarded
    assert payload["rag_variant"] == "V0" and payload["retrieved_chunk_ids"] == []
    assert "ollama down" in payload["retrieval"]["error"]


def test_new_sessions_and_v0_skip_retrieval(env):
    mq, _, monkeypatch = env
    calls = fake_retrieve(monkeypatch, error=AssertionError("must not run"))
    with_session(monkeypatch, None, [])
    route({"prompt": "rescrie textul acesta mai formal te rog frumos", "session_id": str(uuid.uuid4())})
    long_session(monkeypatch)
    settings(monkeypatch, RAG_VARIANT="V0")
    route({"prompt": "rescrie textul acesta mai formal te rog frumos", "session_id": str(uuid.uuid4())})

    assert calls == []
    assert [p["rag_variant"] for _, p in mq.forwarded] == ["V2", "V0"]


def test_a_cut_last_turn_is_completed_by_its_own_chunks(env):
    # one earlier turn, a pasted file far too long for the model: the last turn is cut, and the
    # chunk the question is about comes back as a fragment (retrieval runs from the first turn)
    mq, _, monkeypatch = env
    with_session(monkeypatch, SessionState("text-rewrite", 0.91, 0.6),
                 [TurnText(1, "rescrie fișierul ăsta", LONG_ANSWER)])
    pool = [Fragment(12, 1, 7, "def verify_token(t): return jwt.decode(t, KEY)", "rescrie fișierul ăsta"),
            Fragment(10, 1, 0, "token token token", "rescrie fișierul ăsta")]      # already in the kept part
    calls = fake_retrieve(monkeypatch, router.Retrieved(pool, candidates=20))
    route({"prompt": "explică funcția verify_token din ce ți-am trimis", "session_id": str(uuid.uuid4())})

    ((_, payload),) = mq.forwarded
    assert calls and payload["recent_turns"] == 1
    assert payload["retrieved_chunk_ids"] == [12]
    assert "def verify_token" in payload["messages"][1]["content"]
    assert payload["messages"][-2]["content"].endswith("[…]")             # the cut answer of turn 1


def test_request_variant_needs_the_override_switch(env):
    mq, _, monkeypatch = env
    long_session(monkeypatch)
    calls = fake_retrieve(monkeypatch, router.Retrieved([], candidates=0))
    sid = str(uuid.uuid4())
    route({"prompt": "cum era funcția?", "session_id": sid, "variant": "V1"})       # ignored
    settings(monkeypatch, ALLOW_VARIANT_OVERRIDE="true")
    route({"prompt": "cum era funcția?", "session_id": sid, "variant": "V1"})
    route({"prompt": "cum era funcția?", "session_id": sid, "variant": "V0"})

    assert [c[2] for c in calls] == ["V2", "V1"]
    assert [p["rag_variant"] for _, p in mq.forwarded] == ["V2", "V1", "V0"]
    with pytest.raises(ValueError, match="offline"):
        route({"prompt": "cum era funcția?", "session_id": sid, "variant": "V4"})
