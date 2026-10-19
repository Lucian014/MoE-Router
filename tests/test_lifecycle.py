"""Start-up retries, readiness file and engine health parsing -- no broker needed."""
import asyncio
import socket
import time

import pytest

from engine.ollama_engine import model_is_listed
from shared.lifecycle import Stopped, retry, set_ready
from shared.rabbitmq import RabbitMQClient


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=20))


def flaky(failures: int):
    """An async function that fails `failures` times, then returns 'ok'."""
    calls = {"n": 0}

    async def fn():
        calls["n"] += 1
        if calls["n"] <= failures:
            raise ConnectionError(f"attempt {calls['n']} failed")
        return "ok"

    return fn, calls


# ------------------------------------------------------------------ retry
def test_retry_returns_once_it_succeeds():
    fn, calls = flaky(3)
    assert run(retry(fn, "thing", base_delay=0.01)) == "ok"
    assert calls["n"] == 4


def test_retry_gives_up_with_the_last_error():
    fn, calls = flaky(10)
    with pytest.raises(ConnectionError, match="attempt 3 failed"):
        run(retry(fn, "thing", max_attempts=3, base_delay=0.01))
    assert calls["n"] == 3


def test_retry_backs_off_exponentially_up_to_max_delay(monkeypatch):
    delays = []

    async def fake_sleep(d):
        delays.append(d)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    fn, _ = flaky(6)
    run(retry(fn, "thing", base_delay=1, max_delay=8))
    assert delays == [1, 2, 4, 8, 8, 8]


def test_retry_stops_when_shutdown_is_requested():
    async def scenario():
        stop = asyncio.Event()
        fn, calls = flaky(1000)
        asyncio.get_running_loop().call_later(0.2, stop.set)
        t0 = time.perf_counter()
        with pytest.raises(Stopped):
            await retry(fn, "thing", base_delay=5, stop_event=stop)   # would wait 5 s
        assert time.perf_counter() - t0 < 1
        assert calls["n"] == 1
    run(scenario())


# ------------------------------------------------------------------ readiness file
def test_set_ready(tmp_path):
    f = tmp_path / "ready"
    set_ready(str(f), True)
    assert f.exists()
    set_ready(str(f), True)              # idempotent
    set_ready(str(f), False)
    assert not f.exists()
    set_ready(str(f), False)             # no error when already gone


# ------------------------------------------------------------------ RabbitMQ connect retries
def _closed_port() -> int:
    with socket.socket() as s:           # a port nothing listens on
        s.bind(("localhost", 0))
        return s.getsockname()[1]


def test_connect_gives_up_after_max_retries():
    client = RabbitMQClient("localhost", "u", "p", port=_closed_port())
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="Failed to connect to RabbitMQ"):
        run(client.connect(service_name="test", max_retries=2))   # 1 s backoff in between
    assert 0.9 < time.perf_counter() - t0 < 5


def test_connect_retries_default_comes_from_settings():
    client = RabbitMQClient("localhost", "u", "p", port=_closed_port(), connect_retries=1)
    with pytest.raises(RuntimeError):
        run(client.connect(service_name="test"))


def test_connect_is_aborted_by_the_stop_event():
    async def scenario():
        stop = asyncio.Event()
        client = RabbitMQClient("localhost", "u", "p", port=_closed_port())   # 0 = forever
        asyncio.get_running_loop().call_later(0.3, stop.set)
        with pytest.raises(Stopped):
            await client.connect(service_name="test", stop_event=stop)
    run(scenario())


# ------------------------------------------------------------------ engine health
@pytest.mark.parametrize("model, names, expected", [
    ("gemma3:1b", ["gemma3:1b", "qwen2.5:3b"], True),
    ("gemma3:4b", ["gemma3:1b"], False),
    ("llama3", ["llama3:latest"], True),           # pulled without a tag
    ("llama3:8b", ["llama3:latest"], False),
])
def test_ollama_model_is_listed(model, names, expected):
    assert model_is_listed({"models": [{"name": n} for n in names]}, model) is expected


def test_ollama_model_is_listed_empty_response():
    assert model_is_listed({}, "gemma3:1b") is False
