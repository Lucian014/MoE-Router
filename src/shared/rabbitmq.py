import asyncio
import json
import time
import uuid

import aio_pika

from shared.lifecycle import Stopped, retry

# Header set on error replies, so the caller can tell them apart from a normal result.
STATUS_HEADER = "x-status"
# Absolute deadline of an RPC request (epoch milliseconds, int: AMQP floats are too
# imprecise for epoch values). Set by call(), carried by forward(), checked by consumers,
# so nobody spends GPU time on a request the Gateway has already given up on.
DEADLINE_HEADER = "x-deadline-ms"


class RpcError(Exception):
    """The remote side (router or worker) answered with an error reply."""


class RpcTimeout(Exception):
    """No reply arrived before the deadline."""


def _now_ms() -> int:
    return int(time.time() * 1000)


def remaining_seconds(message: aio_pika.abc.AbstractIncomingMessage) -> float | None:
    """Seconds left before the request's deadline; None if it has none (<= 0: expired)."""
    deadline = (message.headers or {}).get(DEADLINE_HEADER)
    if deadline is None:
        return None
    return (int(deadline) - _now_ms()) / 1000


class RabbitMQClient:
    def __init__(self, host, user, password, port=5672, connect_retries=0):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.connect_retries = connect_retries
        self.connection = None
        self.channel = None
        self.callback_queue = None
        self.futures: dict[str, asyncio.Future] = {}
        self._stop_event = asyncio.Event()
        # consumers started by consume_messages() and the messages they are handling,
        # so drain() can stop new deliveries and wait for the running ones
        self._consumers: list[tuple[aio_pika.abc.AbstractQueue, str]] = []
        self._inflight: set[asyncio.Task] = set()

    @classmethod
    def from_settings(cls, settings):
        """Build a client from shared.config.RabbitMQSettings."""
        return cls(host=settings.host, port=settings.port, user=settings.user,
                   password=settings.password.get_secret_value(), connect_retries=settings.connect_retries)

    async def connect(self, service_name="Service", max_retries=None, stop_event: asyncio.Event | None = None):
        """Connect, retrying with exponential backoff (1, 2, 4 ... 30 s).

        max_retries: attempts before giving up; 0 = forever; None = self.connect_retries
        (RABBITMQ_CONNECT_RETRIES). Once connected, aio-pika's robust connection
        reconnects by itself. Setting `stop_event` (or calling close()) during the retries
        aborts them with shared.lifecycle.Stopped."""
        attempts = self.connect_retries if max_retries is None else max_retries
        self._stop_event.clear()
        if stop_event is not None:
            self._stop_event = stop_event

        async def attempt():
            self.connection = await aio_pika.connect_robust(
                host=self.host, port=self.port, login=self.user, password=self.password
            )
            self.channel = await self.connection.channel()

        print(f"[*] {service_name} - Connecting to RabbitMQ at {self.host}:{self.port}...")
        try:
            await retry(attempt, f"{service_name} - connecting to RabbitMQ", max_attempts=attempts,
                        stop_event=self._stop_event)
        except Stopped:
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to connect to RabbitMQ: {e}") from e
        print(f"[*] {service_name} - Connected successfully!")

    # ------------------------------------------------------------------ RPC client (Gateway)
    async def setup_rpc_client(self):
        """Sets up the exclusive callback queue for receiving responses."""
        self.callback_queue = await self.channel.declare_queue(exclusive=True, auto_delete=True)
        await self.callback_queue.consume(self._on_response, no_ack=True)

    async def _on_response(self, message: aio_pika.IncomingMessage):
        # Late replies (the caller already timed out) have no future any more: drop them.
        future = self.futures.pop(message.correlation_id, None)
        if future is None or future.done():
            return
        try:
            body = json.loads(message.body.decode())
        except Exception as e:
            future.set_exception(RpcError(f"invalid reply: {e}"))
            return
        if (message.headers or {}).get(STATUS_HEADER) == "error":
            future.set_exception(RpcError(body.get("error", "unknown error")))
        else:
            future.set_result(body)

    async def call(self, payload: dict, routing_key: str, timeout: float) -> dict:
        """Publishes a request and waits up to `timeout` seconds for the reply.

        Raises RpcTimeout when no reply arrives in time and RpcError when the router or
        worker answered with an error. The request expires in the queue after `timeout`,
        so it is not processed once nobody waits for it any more."""
        correlation_id = str(uuid.uuid4())
        future = asyncio.get_running_loop().create_future()
        self.futures[correlation_id] = future
        try:
            await self.channel.default_exchange.publish(
                aio_pika.Message(
                    body=json.dumps(payload).encode(),
                    correlation_id=correlation_id,
                    reply_to=self.callback_queue.name,
                    content_type="application/json",
                    expiration=timeout,
                    headers={DEADLINE_HEADER: _now_ms() + int(timeout * 1000)},
                ),
                routing_key=routing_key,
            )
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            raise RpcTimeout(f"no reply within {timeout:g}s") from None
        finally:
            # Always forget the request: on timeout, on error and when the HTTP client
            # disconnects (the task is cancelled), otherwise self.futures grows forever.
            self.futures.pop(correlation_id, None)

    # ------------------------------------------------------------------ replies (Router, Worker)
    async def _reply(self, message: aio_pika.abc.AbstractIncomingMessage, body: dict, error: bool = False):
        if not message.reply_to:
            return
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(body).encode(),
                correlation_id=message.correlation_id,
                content_type="application/json",
                headers={STATUS_HEADER: "error"} if error else None,
            ),
            routing_key=message.reply_to,
        )

    async def reply_error(self, message: aio_pika.abc.AbstractIncomingMessage, error: str):
        """Answer the original caller with an error, so it fails fast instead of timing out."""
        try:
            await self._reply(message, {"error": error, "task_id": message.correlation_id}, error=True)
        except Exception as e:
            print(f"[!] Could not send error reply for {message.correlation_id}: {e}")

    # ------------------------------------------------------------------ consuming
    async def consume_messages(self, queue_name: str, callback, prefetch_count: int = 1):
        """Consume raw messages from a durable queue; `callback(message)` acks them itself.

        Every consumer of this client goes through here, so drain() can stop them and wait
        for the messages they are still handling. `prefetch_count` messages are handled
        concurrently (aio-pika runs each callback in its own task)."""
        await self.channel.set_qos(prefetch_count=prefetch_count)
        queue = await self.channel.declare_queue(queue_name, durable=True)

        async def _tracked(message: aio_pika.IncomingMessage):
            task = asyncio.current_task()
            self._inflight.add(task)
            try:
                await callback(message)
            finally:
                self._inflight.discard(task)

        tag = await queue.consume(_tracked)
        self._consumers.append((queue, tag))
        return queue

    async def drain(self, timeout: float) -> int:
        """Graceful shutdown, step 1: stop receiving new messages, then wait up to `timeout`
        seconds for the in-flight ones. Returns how many were still running.

        Messages that don't finish in time are not lost: they are unacked, so the broker
        redelivers them once the connection closes (another pod, or this one restarted)."""
        for queue, tag in self._consumers:
            try:
                await queue.cancel(tag)
            except Exception as e:
                print(f"[!] Could not cancel consumer on {queue.name}: {e}")
        self._consumers.clear()
        running = set(self._inflight)
        if running:
            print(f"[*] Waiting up to {timeout:g}s for {len(running)} in-flight message(s)...")
            _, pending = await asyncio.wait(running, timeout=timeout)
            if pending:
                print(f"[!] {len(pending)} message(s) still running; they will be redelivered")
            return len(pending)
        return 0

    async def consume_rpc(self, queue_name: str, handler_func, prefetch_count: int = 1):
        """Listens on a durable queue and auto-replies to the caller."""

        async def _wrapper(message: aio_pika.IncomingMessage):
            async with message.process():
                left = remaining_seconds(message)
                if left is not None and left <= 0:
                    print(f"[!] {message.correlation_id}: deadline passed, dropping (caller gave up)")
                    return
                try:
                    payload = json.loads(message.body.decode())
                    result = await handler_func(payload)  # Call custom logic
                except Exception as e:
                    print(f"[!] {message.correlation_id}: handler failed: {e}")
                    await self.reply_error(message, str(e))
                    return
                try:
                    await self._reply(message, result)
                except Exception as e:
                    # e.g. a result that is not JSON-serializable
                    print(f"[!] {message.correlation_id}: could not send reply: {e}")
                    await self.reply_error(message, f"could not send reply: {e}")

        await self.consume_messages(queue_name, _wrapper, prefetch_count=prefetch_count)

    async def forward(self, message: aio_pika.IncomingMessage, routing_key: str,
                      payload: dict | None = None) -> bool:
        """
        Forward an incoming RabbitMQ message to another queue.

        IMPORTANT:
        correlation_id and reply_to are intentionally preserved.

        This is used by the Middleware:

            Gateway -> Middleware -> Worker

        The Worker can therefore reply directly to the Gateway.

        The deadline header is preserved too, and the forwarded message expires when the
        deadline passes. Returns False (nothing forwarded) if the deadline already passed.
        """
        left = remaining_seconds(message)
        if left is not None and left <= 0:
            return False

        body = message.body

        if payload is not None:
            body = json.dumps(payload).encode()

        forwarded_message = aio_pika.Message(
            body=body,

            # Preserve the original RPC correlation ID.
            correlation_id=message.correlation_id,

            # Preserve the Gateway's private callback queue.
            reply_to=message.reply_to,

            # Preserve useful AMQP metadata (including the deadline header).
            content_type=message.content_type,
            content_encoding=message.content_encoding,
            headers=message.headers,

            # Drop the request from the worker queue once the Gateway stopped waiting.
            expiration=left,

            # Make the forwarded request persistent.
            delivery_mode=aio_pika.DeliveryMode.PERSISTENT,
        )

        await self.channel.default_exchange.publish(
            forwarded_message,
            routing_key=routing_key,
        )
        return True

    # ------------------------------------------------------------------ one-way messages
    async def publish(self, payload: dict, routing_key: str):
        """Publishes a message without waiting for a response (Fire-and-forget)."""
        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(payload).encode(),
                content_type="application/json",
            ),
            routing_key=routing_key,
        )

    async def consume(self, queue_name: str, handler_func):
        """Listens on a durable queue for one-way messages (no auto-reply)."""

        async def _wrapper(message: aio_pika.IncomingMessage):
            async with message.process():
                try:
                    payload = json.loads(message.body.decode())
                    await handler_func(payload)
                except Exception as e:
                    print(f"[!] Error processing message: {e}")

        await self.consume_messages(queue_name, _wrapper)

    @property
    def is_connected(self) -> bool:
        return self.channel is not None and not self.channel.is_closed

    async def close(self):
        """Closes the connection (and aborts a connect() still retrying). Call drain() first
        to let in-flight messages finish."""
        self._stop_event.set()
        # Fail requests still waiting for a reply instead of leaving them hanging.
        for future in self.futures.values():
            if not future.done():
                future.set_exception(RpcError("shutting down"))
        self.futures.clear()
        if self.connection:
            try:
                await asyncio.wait_for(self.connection.close(), timeout=5.0)
            except Exception as e:
                print(f"[!] Error closing RabbitMQ connection: {e}")
            finally:
                self.connection = None
                self.channel = None
