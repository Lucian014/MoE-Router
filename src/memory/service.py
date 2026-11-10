"""
memory/service.py -- the memory service: consumes MEMORY_QUEUE and indexes each stored turn.

    Worker --memory.ingest {turn_id}--> memory service --> chunks (Postgres + pgvector)

    python -m memory.service                          # the consumer (the Deployment)
    python -m memory.service --reindex [--force]      # index every turn that needs it, then exit

Robustness (docs/rag-planv2.md 2):
  - the message is acked only after the chunks are committed: a crash mid-way means a redelivery
  - indexing is idempotent (memory/indexer.py), so a redelivery changes nothing
  - a turn that keeps failing (MEMORY_MAX_ATTEMPTS tries, with backoff) is rejected into
    <queue>.dlq instead of blocking the queue; `--reindex` picks it up again later, as it does
    turns whose message was lost (no chunks yet) or that were indexed with older rules

Lifecycle, as the worker: start-up waits for the embedding model and the database; the
readiness file exists while consuming; on SIGTERM it finishes the turn in flight and closes.
"""
import argparse
import asyncio
import json
import sys

from sqlalchemy import text

from database import TurnRepository, dispose_engine, get_session
from memory.indexer import IndexResult, index_turn
from shared.config import get_lifecycle_settings, get_memory_settings, get_rabbitmq_settings
from shared.lifecycle import Stopped, install_stop_signals, retry, set_ready
from shared.memory.chunking import CHUNKER_VERSION
from shared.memory.embedding import EmbeddingClient, OllamaEmbeddingClient
from shared.rabbitmq import RabbitMQClient

RETRY_BASE_SECONDS = 2.0


def make_embedder() -> OllamaEmbeddingClient:
    cfg = get_memory_settings()
    return OllamaEmbeddingClient(cfg.embedding_host, cfg.embedding_model, keep_alive=cfg.embedding_keep_alive)


async def check_ready(embedder: EmbeddingClient) -> None:
    """The embedding model is available and the database answers."""
    await embedder.health()
    async with get_session() as session:
        await session.execute(text("SELECT 1"))


async def handle_message(message, embedder: EmbeddingClient, max_attempts: int) -> None:
    """Ack after the turn is indexed; reject (-> dead-letter queue) a malformed message at once
    and a turn that still fails after max_attempts."""
    try:
        turn_id = int(json.loads(message.body.decode())["turn_id"])
    except Exception as e:
        print(f"[!] malformed message, dead-lettered: {type(e).__name__}: {e}")
        await message.reject(requeue=False)
        return

    for attempt in range(1, max_attempts + 1):
        try:
            result = await index_turn(turn_id, embedder)
        except Exception as e:
            print(f"[!] turn {turn_id}: attempt {attempt}/{max_attempts} failed: {type(e).__name__}: {e}")
            if attempt < max_attempts:
                await asyncio.sleep(RETRY_BASE_SECONDS * 2 ** (attempt - 1))
            continue
        await message.ack()
        _log(result)
        return
    print(f"[!] turn {turn_id}: giving up, dead-lettered (run --reindex once the cause is fixed)")
    await message.reject(requeue=False)


def _log(result: IndexResult) -> None:
    print(f"[OK] turn {result.turn_id}: {result.status}"
          + (f", {result.chunks} chunk(s)" if result.chunks else ""))


async def reindex(force: bool) -> None:
    """One pass over the turns without chunks or with chunks from older rules
    (--force: every turn). Run it in the memory pod: `python -m memory.service --reindex`."""
    embedder = make_embedder()
    await check_ready(embedder)
    async with get_session() as session:
        turns = TurnRepository(session)
        if force:
            pending = await turns.list_ids()
        else:
            pending = [t.id for t in await turns.list_to_index(CHUNKER_VERSION, embedder.model, limit=None)]
    print(f"[*] {len(pending)} turn(s) to index")
    for turn_id in pending:
        try:
            _log(await index_turn(turn_id, embedder, force=force))
        except Exception as e:
            print(f"[!] turn {turn_id}: {type(e).__name__}: {e}")
    print("[OK] reindex finished")
    await dispose_engine()


async def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    rabbit = get_rabbitmq_settings()
    cfg = get_memory_settings()
    lifecycle = get_lifecycle_settings()
    if not rabbit.memory_queue:
        raise SystemExit("MEMORY_QUEUE is empty: nothing to consume")
    embedder = make_embedder()
    mq_client = RabbitMQClient.from_settings(rabbit)
    stop = install_stop_signals()
    set_ready(lifecycle.ready_file, False)

    try:
        await retry(lambda: check_ready(embedder),
                    f"embedding model {embedder.model} and the database", stop_event=stop)
        await mq_client.connect(service_name="Memory", stop_event=stop)
    except Stopped:
        print("[*] stopped before start-up finished")
        return

    arguments = await mq_client.declare_with_dead_letters(rabbit.memory_queue)
    await mq_client.consume_messages(
        rabbit.memory_queue,
        lambda message: handle_message(message, embedder, cfg.max_attempts),
        prefetch_count=1,
        arguments=arguments,
    )
    set_ready(lifecycle.ready_file, True)
    print(f"[*] Memory service listening on {rabbit.memory_queue} (embeddings: {embedder.model} "
          f"on {embedder.host}, chunker v{CHUNKER_VERSION}, dead letters: {rabbit.memory_queue}.dlq)")

    await stop.wait()
    print("[*] SIGTERM received, shutting down")
    set_ready(lifecycle.ready_file, False)
    await mq_client.drain(lifecycle.shutdown_grace_seconds)
    await mq_client.close()
    await dispose_engine()
    print("[OK] memory service stopped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reindex", action="store_true", help="index the turns that need it, then exit")
    parser.add_argument("--force", action="store_true", help="with --reindex: reindex every turn")
    args = parser.parse_args()
    asyncio.run(reindex(args.force) if args.reindex else main())
