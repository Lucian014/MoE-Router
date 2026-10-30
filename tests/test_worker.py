"""The worker's request handling (src/worker/worker.py), with a fake engine instead of Ollama."""
import asyncio

import pytest

from engine import ChatResult
from shared.prompts import SYSTEM_PROMPT
from worker import worker


class FakeEngine:
    def __init__(self, result=ChatResult("ok", 10, 2)):
        self.result, self.calls = result, []

    async def chat(self, model, messages, max_tokens, num_ctx):
        self.calls.append(dict(model=model, messages=messages, max_tokens=max_tokens, num_ctx=num_ctx))
        return self.result


@pytest.fixture
def fake(monkeypatch):
    engine = FakeEngine()
    monkeypatch.setattr(worker, "engine", engine)
    monkeypatch.setattr(worker, "MODEL_NAME", "gemma3:1b")
    monkeypatch.setattr(worker, "ENGINE_TYPE", "ollama")
    monkeypatch.setattr(worker, "CONTEXT_WINDOW", 4096)
    monkeypatch.setattr(worker, "MAX_OUTPUT_TOKENS", 512)
    return engine


def handle(payload):
    return asyncio.run(worker.handle_request(payload))


def test_plain_prompt_gets_the_system_prompt_and_default_limits(fake):
    reply = handle({"prompt": "salut", "model": "gemma3:1b", "category": "open-qa"})

    (call,) = fake.calls
    assert call["messages"] == [{"role": "system", "content": SYSTEM_PROMPT},
                                {"role": "user", "content": "salut"}]
    assert (call["num_ctx"], call["max_tokens"]) == (4096, 512)
    assert reply["response"] == "ok"
    assert (reply["prompt_tokens"], reply["completion_tokens"], reply["truncated"]) == (10, 2, False)
    assert reply["routing"] == {"category": "open-qa"}          # consumed fields are not echoed back


def test_messages_and_limits_from_the_router_are_used_as_given(fake):
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "q1"},
                {"role": "assistant", "content": "a1"}, {"role": "user", "content": "q2"}]
    reply = handle({"messages": messages, "num_ctx": 8192, "max_tokens": 1024, "turn": 2,
                    "task_type": "coding"})

    (call,) = fake.calls
    assert call["messages"] == messages
    assert (call["num_ctx"], call["max_tokens"]) == (8192, 1024)
    assert reply["routing"] == {"task_type": "coding"}


@pytest.mark.parametrize("payload, error", [
    ({}, "needs 'messages' or a non-empty 'prompt'"),
    ({"prompt": ""}, "needs 'messages' or a non-empty 'prompt'"),
    ({"messages": []}, "non-empty list"),
    ({"messages": [{"role": "tool", "content": "x"}]}, "invalid message"),
    ({"messages": [{"role": "user", "content": 3}]}, "invalid message"),
    ({"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]}, "last message"),
    ({"prompt": "q", "num_ctx": 512, "max_tokens": 512}, "max_tokens < num_ctx"),
])
def test_invalid_payloads_are_rejected_before_the_engine(fake, payload, error):
    with pytest.raises(ValueError, match=error):
        handle(payload)
    assert fake.calls == []


# ------------------------------------------------------------------ storing the turn
def test_without_session_id_nothing_is_stored(fake):
    assert handle({"prompt": "salut"})["turn"] is None          # store_turn returns before the DB


def test_turn_is_stored_with_the_original_prompt(fake, monkeypatch):
    stored = []

    async def store(payload, user_content, result):
        stored.append((payload["session_id"], user_content, result.text))
        return 3

    monkeypatch.setattr(worker, "store_turn", store)
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "context + q"}]
    reply = handle({"prompt": "q", "messages": messages, "session_id": "sid"})

    assert stored == [("sid", "q", "ok")]                       # the user's prompt, not the context
    assert reply["turn"] == 3


@pytest.mark.parametrize("error", [RuntimeError("db down"), worker.TurnAlreadyExistsError("sid", 1)])
def test_a_failed_store_still_answers(fake, monkeypatch, error):
    async def store(*_):
        raise error

    monkeypatch.setattr(worker, "store_turn", store)
    reply = handle({"prompt": "q", "session_id": "sid"})
    assert (reply["response"], reply["turn"]) == ("ok", None)
