"""The generic worker (src/worker/worker.py): request handling with a fake engine instead of Ollama,
and following the registry's enabled models with a fake RabbitMQ client."""
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


MODEL = None


@pytest.fixture
def fake(monkeypatch):
    global MODEL
    engine = FakeEngine()
    MODEL = worker.ServedModel(model_id="gemma3:1b", engine_type="ollama", host="http://ollama:11434",
                               context_window=4096, max_output_tokens=512, engine=engine)
    return engine


def handle(payload):
    return asyncio.run(worker.handle_request(payload, MODEL))


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


class FakeMQ:
    def __init__(self, fail=False):
        self.published, self.fail = [], fail

    async def publish(self, payload, routing_key, persistent=False):
        if self.fail:
            raise ConnectionError("broker gone")
        self.published.append((routing_key, payload, persistent))


def stored_as(turn, turn_id, log):
    async def store(payload, user_content, result, model_id):
        log.append((payload["session_id"], user_content, result.text, model_id))
        return turn, turn_id
    return store


def test_turn_is_stored_with_the_original_prompt_and_sent_for_indexing(fake, monkeypatch):
    stored, mq = [], FakeMQ()
    monkeypatch.setattr(worker, "store_turn", stored_as(3, 42, stored))
    monkeypatch.setattr(worker, "mq_client", mq)
    monkeypatch.setattr(worker, "MEMORY_QUEUE", "memory.ingest")
    messages = [{"role": "system", "content": "s"}, {"role": "user", "content": "context + q"}]
    reply = handle({"prompt": "q", "messages": messages, "session_id": "sid"})

    assert stored == [("sid", "q", "ok", "gemma3:1b")]          # the user's prompt, not the context
    assert reply["turn"] == 3
    assert mq.published == [("memory.ingest", {"turn_id": 42}, True)]


def test_nothing_is_published_without_a_memory_queue(fake, monkeypatch):
    mq = FakeMQ()
    monkeypatch.setattr(worker, "store_turn", stored_as(1, 7, []))
    monkeypatch.setattr(worker, "mq_client", mq)
    monkeypatch.setattr(worker, "MEMORY_QUEUE", "")
    assert handle({"prompt": "q", "session_id": "sid"})["turn"] == 1
    assert mq.published == []


def test_a_failed_publish_still_answers(fake, monkeypatch):
    monkeypatch.setattr(worker, "store_turn", stored_as(1, 7, []))
    monkeypatch.setattr(worker, "mq_client", FakeMQ(fail=True))
    monkeypatch.setattr(worker, "MEMORY_QUEUE", "memory.ingest")
    reply = handle({"prompt": "q", "session_id": "sid"})
    assert (reply["response"], reply["turn"]) == ("ok", 1)


@pytest.mark.parametrize("error", [RuntimeError("db down"), worker.TurnAlreadyExistsError("sid", 1)])
def test_a_failed_store_still_answers(fake, monkeypatch, error):
    async def store(*_):
        raise error

    monkeypatch.setattr(worker, "store_turn", store)
    reply = handle({"prompt": "q", "session_id": "sid"})
    assert (reply["response"], reply["turn"]) == ("ok", None)


def test_the_queue_decides_the_model(fake):
    reply = handle({"prompt": "q", "model": "another:7b"})
    assert fake.calls[0]["model"] == "gemma3:1b"
    assert (reply["model"], reply["engine"]) == ("gemma3:1b", "ollama")


# ------------------------------------------------------------------ following the registry
def row(model_id, engine="ollama", host=None, context_window=4096, max_output_tokens=512):
    return {"model_id": model_id, "engine": engine, "inference_host": host,
            "context_window": context_window, "max_output_tokens": max_output_tokens}


def test_a_model_without_host_runs_on_the_default_ollama():
    m = worker.served_from_row(row("gemma3:1b"), "http://ollama:11434")
    assert (m.host, m.problem, m.queue_name) == ("http://ollama:11434", None, "worker.gemma3-1b")
    assert m.engine.host == "http://ollama:11434"


@pytest.mark.parametrize("r, problem", [
    (row("qwen-32b", engine="vllm"), "needs an inference_host"),
    (row("x", engine="tgi", host="http://h"), "Unknown engine"),
])
def test_a_model_that_cannot_be_served_says_why(r, problem):
    m = worker.served_from_row(r, "http://ollama:11434")
    assert m.engine is None and problem in m.problem


class HealthEngine:
    """Answers health checks while `up`."""
    def __init__(self, host, up=True):
        self.host, self.up = host, up

    async def health(self, model):
        if not self.up:
            raise RuntimeError(f"model '{model}' is not pulled")


class SyncMQ:
    """Records what the worker consumes; no broker."""
    def __init__(self):
        self.declared, self.consuming, self.channels = [], {}, 0

    async def declare_queue(self, name, arguments=None):
        self.declared.append(name)

    async def open_channel(self):
        self.channels += 1
        return FakeChannel()

    async def consume_rpc(self, queue_name, handler, prefetch_count=1, channel=None):
        self.consuming[queue_name] = (handler, prefetch_count, channel)
        return queue_name, f"tag-{queue_name}"

    async def cancel_consumer(self, queue, tag):
        self.consuming.pop(queue, None)


class FakeChannel:
    closed = False

    async def close(self):
        self.closed = True


@pytest.fixture
def registry(monkeypatch):
    """The enabled models (edit the list between syncs) and the engines built for them."""
    rows, engines = [], {}

    async def load():
        return [dict(r) for r in rows]

    def get_engine(engine_type, host):
        if engine_type not in ("ollama", "vllm"):
            raise ValueError(f"Unknown engine '{engine_type}'")
        return engines.setdefault(host, HealthEngine(host))

    mq = SyncMQ()
    monkeypatch.setattr(worker, "load_enabled_models", load)
    monkeypatch.setattr(worker, "get_engine", get_engine)
    monkeypatch.setattr(worker, "mq_client", mq)
    monkeypatch.setattr(worker, "MODELS", {})
    monkeypatch.setattr(worker, "OLLAMA_URL", "http://ollama:11434")
    monkeypatch.setattr(worker, "PREFETCH", 2)
    monkeypatch.setattr(worker, "SHUTDOWN_GRACE_SECONDS", 0)
    return rows, engines, mq


def sync():
    async def run():
        await worker.sync_models()
        await asyncio.gather(*worker._background)       # channels closed once idle
    asyncio.run(run())


def test_every_enabled_model_that_answers_is_consumed_on_its_own_channel(registry):
    rows, _, mq = registry
    rows += [row("gemma3:1b"), row("qwen-32b", engine="vllm", host="http://a100:8000")]
    sync()
    assert set(mq.consuming) == {"worker.gemma3-1b", "worker.qwen-32b"}
    assert mq.channels == 2                                       # one channel per model
    assert {prefetch for _, prefetch, _ in mq.consuming.values()} == {2}


def test_a_model_whose_engine_stops_answering_is_paused_and_resumed(registry):
    rows, engines, mq = registry
    rows += [row("gemma3:1b"), row("qwen-32b", engine="vllm", host="http://a100:8000")]
    sync()
    engines["http://a100:8000"].up = False
    sync()
    assert set(mq.consuming) == {"worker.gemma3-1b"}            # the other model keeps going
    engines["http://a100:8000"].up = True
    sync()
    assert set(mq.consuming) == {"worker.gemma3-1b", "worker.qwen-32b"}
    assert mq.channels == 2                                       # the paused model kept its channel


def test_a_model_that_never_answered_still_gets_its_queue_declared(registry):
    rows, _, mq = registry
    rows.append(row("qwen-32b", engine="vllm"))                   # no inference_host
    sync()
    assert mq.consuming == {}
    assert mq.declared == ["worker.qwen-32b"]                     # requests wait instead of being dropped


def test_a_disabled_model_is_retired(registry):
    rows, _, mq = registry
    rows += [row("gemma3:1b"), row("llama3.2:3b")]
    sync()
    channel = worker.MODELS["llama3.2:3b"].channel
    rows.pop()
    sync()
    assert set(mq.consuming) == {"worker.gemma3-1b"}
    assert "llama3.2:3b" not in worker.MODELS and channel.closed


def test_a_model_moved_to_another_server_gets_a_new_engine(registry):
    rows, _, mq = registry
    rows.append(row("qwen-32b", engine="vllm", host="http://a100:8000"))
    sync()
    rows[0] = row("qwen-32b", engine="vllm", host="http://other:8000")
    sync()
    assert worker.MODELS["qwen-32b"].engine.host == "http://other:8000"
    assert "worker.qwen-32b" in mq.consuming and mq.channels == 2


def test_new_limits_apply_without_a_new_consumer(registry):
    rows, _, mq = registry
    rows.append(row("gemma3:1b"))
    sync()
    rows[0] = row("gemma3:1b", context_window=8192, max_output_tokens=1024)
    sync()
    m = worker.MODELS["gemma3:1b"]
    assert (m.context_window, m.max_output_tokens) == (8192, 1024)
    assert mq.channels == 1
