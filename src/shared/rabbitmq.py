import asyncio
import json
import uuid
import aio_pika


class RabbitMQClient:
    def __init__(self, host, user, password):
        self.host = host
        self.user = user
        self.password = password
        self.connection = None
        self.channel = None
        self.callback_queue = None
        self.futures = {}
        self._should_stop = False
        self._stop_event = asyncio.Event()

    async def connect(self, service_name="Service", max_retries=10):
        self._should_stop = False
        self._stop_event.clear()
        for attempt in range(max_retries):
            if self._should_stop or self._stop_event.is_set():
                break
            try:
                print(f"[*] {service_name} - Connecting to RabbitMQ ({attempt + 1}/{max_retries})...")
                self.connection = await aio_pika.connect_robust(
                    host=self.host, login=self.user, password=self.password
                )
                self.channel = await self.connection.channel()
                print(f"[*] {service_name} - Connected successfully!")
                return
            except Exception as e:
                print(f"[!] Connection failed: {e}. Retrying in 5s...")
                try:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=5.0)
                except asyncio.TimeoutError:
                    pass
                if self._should_stop or self._stop_event.is_set():
                    break
        raise RuntimeError("Failed to connect to RabbitMQ.")

    async def setup_rpc_client(self):
        """Sets up the exclusive callback queue for receiving responses."""
        self.callback_queue = await self.channel.declare_queue(exclusive=True, auto_delete=True)
        await self.callback_queue.consume(self._on_response, no_ack=True)

    async def _on_response(self, message: aio_pika.IncomingMessage):
        future = self.futures.pop(message.correlation_id, None)
        if future:
            future.set_result(json.loads(message.body.decode()))

    async def call(self, payload: dict, routing_key: str) -> dict:
        """Publishes a message and waits for the response."""
        correlation_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self.futures[correlation_id] = future

        await self.channel.default_exchange.publish(
            aio_pika.Message(
                body=json.dumps(payload).encode(),
                correlation_id=correlation_id,
                reply_to=self.callback_queue.name,
                content_type="application/json",
            ),
            routing_key=routing_key,
        )
        return await future

    async def consume_rpc(self, queue_name: str, handler_func):
        """Listens on a durable queue and auto-replies to the caller."""
        await self.channel.set_qos(prefetch_count=1)
        queue = await self.channel.declare_queue(queue_name, durable=True)

        async def _wrapper(message: aio_pika.IncomingMessage):
            async with message.process():
                try:
                    payload = json.loads(message.body.decode())
                    result = await handler_func(payload)  # Call your custom logic

                    if message.reply_to:
                        await self.channel.default_exchange.publish(
                            aio_pika.Message(
                                body=json.dumps(result).encode(),
                                correlation_id=message.correlation_id,
                            ),
                            routing_key=message.reply_to,
                        )
                except Exception as e:
                    if message.reply_to:
                        error_msg = {"error": str(e), "task_id": message.correlation_id}
                        await self.channel.default_exchange.publish(
                            aio_pika.Message(
                                body=json.dumps(error_msg).encode(),
                                correlation_id=message.correlation_id,
                            ),
                            routing_key=message.reply_to,
                        )

        await queue.consume(_wrapper)

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
        await self.channel.set_qos(prefetch_count=1)
        queue = await self.channel.declare_queue(queue_name, durable=True)

        async def _wrapper(message: aio_pika.IncomingMessage):
            async with message.process():
                try:
                    payload = json.loads(message.body.decode())
                    await handler_func(payload)
                except Exception as e:
                    print(f"[!] Error processing message: {e}")

        await queue.consume(_wrapper)

    async def close(self):
        self._should_stop = True
        self._stop_event.set()
        if self.connection:
            try:
                await asyncio.wait_for(self.connection.close(), timeout=5.0)
            except Exception as e:
                print(f"[!] Error closing RabbitMQ connection: {e}")
            finally:
                self.connection = None
                self.channel = None