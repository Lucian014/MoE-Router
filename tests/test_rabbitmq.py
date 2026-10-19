"""
RPC over RabbitMQ (src/shared/rabbitmq.py) against a real broker: Gateway -> Router -> Worker.

Needs a RabbitMQ; skipped unless RABBITMQ_TEST_PORT is set. Locally:

    podman run -d --rm --name rpc-test-rabbit -p 56720:5672 \\
        -e RABBITMQ_DEFAULT_USER=test -e RABBITMQ_DEFAULT_PASS=test docker.io/library/rabbitmq:3.13
    RABBITMQ_TEST_PORT=56720 pytest tests/test_rabbitmq.py

Optional: RABBITMQ_TEST_HOST (localhost), RABBITMQ_TEST_USER / RABBITMQ_TEST_PASS (test / test).
"""
import asyncio
import os
import time
import uuid

import aio_pika
import pytest

from shared.rabbitmq import DEADLINE_HEADER, RabbitMQClient, RpcError, RpcTimeout, remaining_seconds

pytestmark = [
    pytest.mark.rabbitmq,
    pytest.mark.skipif("RABBITMQ_TEST_PORT" not in os.environ, reason="RABBITMQ_TEST_PORT not set"),
]


def _client():
    return RabbitMQClient(
        host=os.getenv("RABBITMQ_TEST_HOST", "localhost"),
        port=int(os.getenv("RABBITMQ_TEST_PORT", "5672")),
        user=os.getenv("RABBITMQ_TEST_USER", "test"),
        password=os.getenv("RABBITMQ_TEST_PASS", "test"),
    )


class Pipeline:
    """A Gateway, a Router that forwards (or fails on 'route-fail'), and a Worker running `handler`.
    Queue names are unique per test, and the queues are deleted afterwards."""

    def __init__(self, prefetch=1):
        self.prefetch = prefetch
        tag = uuid.uuid4().hex[:8]
        self.router_queue, self.worker_queue = f"test.router.{tag}", f"test.worker.{tag}"
        self.calls = []
        self.gateway, self.router, self.worker = _client(), _client(), _client()

    async def handler(self, payload):
        self.calls.append(payload)
        if payload.get("boom"):
            raise ValueError("engine exploded")
        if payload.get("sleep"):
            await asyncio.sleep(payload["sleep"])
        return {"echo": payload.get("prompt")}

    async def __aenter__(self):
        for c, name in [(self.gateway, "gateway"), (self.router, "router"), (self.worker, "worker")]:
            await c.connect(service_name=name, max_retries=1)
        await self.gateway.setup_rpc_client()
        await self.worker.consume_rpc(self.worker_queue, self.handler, prefetch_count=self.prefetch)

        await self.router.channel.set_qos(prefetch_count=10)
        queue = await self.router.channel.declare_queue(self.router_queue, durable=True)

        async def route(message):
            async with message.process():
                try:
                    if b"route-fail" in message.body:
                        raise RuntimeError("no model for category")
                    await self.router.forward(message, self.worker_queue)
                except Exception as e:
                    await self.router.reply_error(message, f"routing failed: {e}")

        await queue.consume(route)
        return self

    def call(self, payload, timeout=5, routing_key=None):
        return self.gateway.call(payload, routing_key or self.router_queue, timeout=timeout)

    async def __aexit__(self, *exc):
        cleanup = _client()
        await cleanup.connect(service_name="cleanup", max_retries=1)
        for c in (self.gateway, self.router, self.worker):
            if c.connection:
                await c.close()
        for name in (self.router_queue, self.worker_queue):
            await cleanup.channel.queue_delete(name)
        await cleanup.close()


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=30))


def test_happy_path():
    async def scenario():
        async with Pipeline() as p:
            assert await p.call({"prompt": "hi"}) == {"echo": "hi"}
            assert not p.gateway.futures
    run(scenario())


def test_worker_error_fails_fast_with_rpc_error():
    async def scenario():
        async with Pipeline() as p:
            t0 = time.perf_counter()
            with pytest.raises(RpcError, match="engine exploded"):
                await p.call({"prompt": "x", "boom": True})
            assert time.perf_counter() - t0 < 1
    run(scenario())


def test_router_error_fails_fast_with_rpc_error():
    async def scenario():
        async with Pipeline() as p:
            with pytest.raises(RpcError, match="routing failed: no model for category"):
                await p.call({"prompt": "route-fail"})
            assert p.calls == []                     # never reached the worker
    run(scenario())


def test_timeout_when_nobody_consumes_and_request_expires_from_queue():
    async def scenario():
        async with Pipeline() as p:
            nobody = f"test.nobody.{uuid.uuid4().hex[:8]}"
            await p.gateway.channel.declare_queue(nobody, durable=True)
            t0 = time.perf_counter()
            with pytest.raises(RpcTimeout):
                await p.call({"prompt": "x"}, timeout=1, routing_key=nobody)
            assert 0.9 < time.perf_counter() - t0 < 1.5
            assert not p.gateway.futures
            await asyncio.sleep(0.5)
            queue = await p.gateway.channel.declare_queue(nobody, passive=True)
            assert queue.declaration_result.message_count == 0
            await p.gateway.channel.queue_delete(nobody)
    run(scenario())


def test_late_reply_after_timeout_is_ignored():
    async def scenario():
        async with Pipeline() as p:
            with pytest.raises(RpcTimeout):
                await p.call({"prompt": "slow", "sleep": 1.5}, timeout=0.5)
            await asyncio.sleep(1.5)                 # the late reply arrives now
            assert not p.gateway.futures
            assert await p.call({"prompt": "next"}) == {"echo": "next"}
    run(scenario())


def test_prefetch_runs_requests_concurrently():
    async def scenario():
        async with Pipeline(prefetch=3) as p:
            t0 = time.perf_counter()
            results = await asyncio.gather(*[p.call({"prompt": f"p{i}", "sleep": 1}) for i in range(3)])
            assert time.perf_counter() - t0 < 1.8
            assert sorted(r["echo"] for r in results) == ["p0", "p1", "p2"]
    run(scenario())


def test_cancelled_call_leaves_no_pending_future():
    async def scenario():
        async with Pipeline() as p:
            task = asyncio.create_task(p.call({"prompt": "c", "sleep": 0.5}))
            await asyncio.sleep(0.1)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert not p.gateway.futures
            await asyncio.sleep(0.6)
    run(scenario())


def test_close_fails_pending_calls():
    async def scenario():
        async with Pipeline() as p:
            pending = asyncio.create_task(p.call({"prompt": "x", "sleep": 5}, timeout=10))
            await asyncio.sleep(0.2)
            await p.gateway.close()
            with pytest.raises(RpcError, match="shutting down"):
                await pending
    run(scenario())


def test_stale_request_is_dropped_without_running_the_handler():
    async def scenario():
        async with Pipeline() as p:
            await p.gateway.channel.default_exchange.publish(
                aio_pika.Message(b'{"prompt": "stale"}',
                                 headers={DEADLINE_HEADER: int(time.time() * 1000) - 1000}),
                routing_key=p.worker_queue,
            )
            await asyncio.sleep(0.5)
            assert p.calls == []
    run(scenario())


def test_remaining_seconds():
    class Message:
        headers = {}
    assert remaining_seconds(Message) is None
    Message.headers = {DEADLINE_HEADER: int(time.time() * 1000) + 2000}
    assert 1.5 < remaining_seconds(Message) <= 2
    Message.headers = {DEADLINE_HEADER: int(time.time() * 1000) - 1}
    assert remaining_seconds(Message) <= 0
