"""
Session memory tables (sessions, turns, chunks) and the context limits on `models`, against a real
Postgres with pgvector. The migrations are applied first, so this also checks that they run.

Skipped unless POSTGRES_TEST_PORT is set. Locally:

    podman run -d --rm --name memory-test-pg -p 55432:5432 -e POSTGRES_USER=test \\
        -e POSTGRES_PASSWORD=test -e POSTGRES_DB=model_registry docker.io/pgvector/pgvector:pg16
    POSTGRES_TEST_PORT=55432 pytest tests/test_database_memory.py

Optional: POSTGRES_TEST_HOST (localhost), POSTGRES_TEST_USER / POSTGRES_TEST_PASSWORD (test / test),
POSTGRES_TEST_DB (model_registry). Every test works in its own session id and deletes it afterwards.
"""
import asyncio
import os
import subprocess
import sys
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from database import (
    ChatSessionRepository,
    ChunkRepository,
    ModelRepository,
    TurnAlreadyExistsError,
    TurnRepository,
)
from database.models.chunk.schema import EMBEDDING_DIM

pytestmark = [
    pytest.mark.postgres,
    pytest.mark.skipif("POSTGRES_TEST_PORT" not in os.environ, reason="POSTGRES_TEST_PORT not set"),
]

DATABASE_DIR = Path(__file__).resolve().parents[1] / "src" / "database"


def _env() -> dict[str, str]:
    return {
        "POSTGRES_HOST": os.getenv("POSTGRES_TEST_HOST", "localhost"),
        "POSTGRES_PORT": os.getenv("POSTGRES_TEST_PORT", "5432"),
        "POSTGRES_USER": os.getenv("POSTGRES_TEST_USER", "test"),
        "POSTGRES_PASSWORD": os.getenv("POSTGRES_TEST_PASSWORD", "test"),
        "POSTGRES_DB": os.getenv("POSTGRES_TEST_DB", "model_registry"),
    }


@pytest.fixture(scope="module", autouse=True)
def migrated():
    """alembic upgrade head, in a subprocess so the app's cached settings are not touched."""
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=DATABASE_DIR,
                   env={**os.environ, **_env()}, check=True, capture_output=True)


def run(coro):
    return asyncio.run(asyncio.wait_for(coro, timeout=30))


@asynccontextmanager
async def db():
    """A session on its own engine; commits on success like database.session.get_session()."""
    e = _env()
    engine = create_async_engine(URL.create(
        "postgresql+asyncpg", username=e["POSTGRES_USER"], password=e["POSTGRES_PASSWORD"],
        host=e["POSTGRES_HOST"], port=int(e["POSTGRES_PORT"]), database=e["POSTGRES_DB"]))
    session = async_sessionmaker(engine, expire_on_commit=False)()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()
        await engine.dispose()


@pytest.fixture
def sid():
    """A fresh session id, deleted (with its turns and chunks) after the test."""
    session_id = uuid.uuid4()
    yield session_id

    async def cleanup():
        async with db() as s:
            await ChatSessionRepository(s).delete(session_id)
    run(cleanup())


def onehot(i: int) -> list[float]:
    v = [0.0] * EMBEDDING_DIM
    v[i] = 1.0
    return v


def chunk_row(sid, turn_id, turn, ord_, vec, **over):
    row = dict(turn_id=turn_id, session_id=sid, turn=turn, ord=ord_, source="response", kind="prose",
               content=f"chunk {turn}.{ord_}", content_hash=f"h{turn}{ord_}", tokens=10,
               terms=["chunk", f"t{turn}"], embedding=vec, embedding_model="bge-m3", chunker_version="1")
    row.update(over)
    return row


async def add_turns(sid, n):
    async with db() as s:
        await ChatSessionRepository(s).ensure(sid)
        turns = TurnRepository(s)
        return [await turns.create(sid, t, f"q{t}", f"a{t}") for t in range(1, n + 1)]


# ------------------------------------------------------------------ sessions and turns
def test_recent_turns_come_back_oldest_first(sid):
    async def go():
        await add_turns(sid, 3)
        async with db() as s:
            turns = TurnRepository(s)
            recent = await turns.list_recent(sid, limit=2)
            assert [t.turn for t in recent] == [2, 3]
            assert await turns.last_turn_number(sid) == 3
            assert await turns.last_turn_number(uuid.uuid4()) == 0      # unknown session = empty
    run(go())


def test_ensure_is_idempotent_and_routing_state_is_stored(sid):
    async def go():
        async with db() as s:
            repo = ChatSessionRepository(s)
            await repo.ensure(sid, user_id="u1")
            await repo.ensure(sid, user_id="ignored")                   # second call: no error, no change
            await repo.update_routing_state(sid, "coding", 0.91, 0.84)
        async with db() as s:
            row = await ChatSessionRepository(s).get(sid)
            assert (row.user_id, row.domain) == ("u1", "coding")
            assert row.complexity_score == pytest.approx(0.84)
    run(go())


def test_duplicate_turn_number_raises_without_breaking_the_transaction(sid):
    async def go():
        await add_turns(sid, 1)
        async with db() as s:
            turns = TurnRepository(s)
            with pytest.raises(TurnAlreadyExistsError):
                await turns.create(sid, 1, "again", "again")
            assert await turns.last_turn_number(sid) == 1                # session still usable
    run(go())


def test_turn_create_rejects_unknown_columns(sid):
    async def go():
        async with db() as s:
            await ChatSessionRepository(s).ensure(sid)
            with pytest.raises(ValueError, match="no_such_column"):
                await TurnRepository(s).create(sid, 1, "q", "a", no_such_column=1)
    run(go())


# ------------------------------------------------------------------ chunks
def test_chunk_insert_is_idempotent(sid):
    async def go():
        (t1,) = await add_turns(sid, 1)
        rows = [chunk_row(sid, t1, 1, 0, onehot(0)), chunk_row(sid, t1, 1, 1, onehot(1))]
        async with db() as s:
            assert await ChunkRepository(s).insert_many(rows) == 2
        async with db() as s:
            assert await ChunkRepository(s).insert_many(rows) == 0      # redelivered message
            assert [c.ord for c in await ChunkRepository(s).list_for_turn(t1)] == [0, 1]
    run(go())


def test_candidates_carry_distance_and_skip_newer_turns(sid):
    async def go():
        t1, t2, _ = await add_turns(sid, 3)
        async with db() as s:
            await ChunkRepository(s).insert_many([
                chunk_row(sid, t1, 1, 0, onehot(0)),
                chunk_row(sid, t1, 1, 1, onehot(1)),
                chunk_row(sid, t2, 2, 0, onehot(2)),
            ])
        async with db() as s:
            chunks = ChunkRepository(s)
            got = {(c.turn, c.ord): c for c in await chunks.list_candidates(sid, 3, onehot(0))}
            assert set(got) == {(1, 0), (1, 1), (2, 0)}
            assert got[(1, 0)].distance == pytest.approx(0.0)           # same direction
            assert got[(1, 1)].distance == pytest.approx(1.0)           # orthogonal
            assert got[(1, 0)].terms == ["chunk", "t1"]

            older = await chunks.list_candidates(sid, 2, onehot(0))     # turn 2 is excluded
            assert {c.turn for c in older} == {1}

            no_dense = await chunks.list_candidates(sid, 3, None)       # BM25 only
            assert all(c.distance is None for c in no_dense)

            contents = await chunks.get_contents([got[(2, 0)].id])
            assert contents[got[(2, 0)].id].content == "chunk 2.0"
    run(go())


def test_contextual_candidates_only_include_contextualized_chunks(sid):
    async def go():
        (t1,) = await add_turns(sid, 1)
        async with db() as s:
            await ChunkRepository(s).insert_many([chunk_row(sid, t1, 1, 0, onehot(0)),
                                                  chunk_row(sid, t1, 1, 1, onehot(1))])
        async with db() as s:
            chunks = ChunkRepository(s)
            first = (await chunks.list_for_turn(t1))[0]
            await chunks.set_context(first.id, "context", ["ctx"], onehot(5), "qwen2.5:3b")
        async with db() as s:
            got = await ChunkRepository(s).list_candidates(sid, 2, onehot(5), contextual=True)
            assert [(c.ord, c.terms) for c in got] == [(0, ["ctx"])]
            assert got[0].distance == pytest.approx(0.0)
    run(go())


def test_check_constraints_reject_bad_chunks(sid):
    async def go():
        (t1,) = await add_turns(sid, 1)
        for bad in ({"kind": "image"}, {"source": "system"}, {"tokens": 0}):
            with pytest.raises(IntegrityError):
                async with db() as s:
                    await ChunkRepository(s).insert_many([chunk_row(sid, t1, 1, 0, onehot(0), **bad)])
    run(go())


def test_turns_to_index(sid):
    async def go():
        t1, t2, t3 = await add_turns(sid, 3)
        async with db() as s:
            await ChunkRepository(s).insert_many([
                chunk_row(sid, t2, 2, 0, onehot(0)),                                  # up to date
                chunk_row(sid, t3, 3, 0, onehot(0), chunker_version="0"),             # old chunker
            ])
        async with db() as s:
            pending = {t.id for t in await TurnRepository(s).list_to_index("1", "bge-m3", limit=10_000)}
            assert {t1, t3} <= pending and t2 not in pending
    run(go())


def test_deleting_a_session_cascades(sid):
    async def go():
        (t1,) = await add_turns(sid, 1)
        async with db() as s:
            await ChunkRepository(s).insert_many([chunk_row(sid, t1, 1, 0, onehot(0))])
            await ChatSessionRepository(s).delete(sid)
        async with db() as s:
            left = await s.execute(text("SELECT (SELECT count(*) FROM turns WHERE session_id = :s)"
                                        " + (SELECT count(*) FROM chunks WHERE session_id = :s)"),
                                   {"s": sid})
            assert left.scalar_one() == 0
    run(go())


# ------------------------------------------------------------------ models: context limits
def test_model_context_limits():
    model_id = f"test-{uuid.uuid4().hex[:8]}"

    async def go():
        try:
            async with db() as s:
                repo = ModelRepository(s)
                entry = await repo.create(model_id, "ollama", enabled=False)
                assert (entry.context_window, entry.max_output_tokens) == (4096, 512)
                with pytest.raises(ValueError):
                    await repo.update(model_id, max_output_tokens=4096)          # no room for the prompt
                updated = await repo.update(model_id, context_window=8192, max_output_tokens=1024)
                assert updated.to_dict()["context_window"] == 8192
            with pytest.raises(IntegrityError):                                 # the DB enforces it too
                async with db() as s:
                    await s.execute(text("UPDATE models SET max_output_tokens = 0 WHERE model_id = :m"),
                                    {"m": model_id})
        finally:
            async with db() as s:
                await s.execute(text("DELETE FROM models WHERE model_id = :m"), {"m": model_id})
    run(go())
