from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from sqlalchemy import delete, literal, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.chunk.schema import Chunk


@dataclass(frozen=True)
class ChunkCandidate:
    """What R2 returns per chunk: enough to rank (dense distance + BM25 terms), deduplicate and
    budget, but neither the vector nor the content (docs/rag-planv2.md, 4.6)."""

    id: int
    turn: int
    ord: int
    kind: str
    tokens: int
    content_hash: str
    terms: list[str]
    distance: float | None      # cosine distance to the query, None when dense search is off


class ChunkRepository:
    """Data access for `chunks`. Only flushes; get_session() owns the transaction."""

    def __init__(self, session: AsyncSession):
        self._session = session

    # -- writes (memory service) -------------------------------------------

    async def insert_many(self, rows: Sequence[dict[str, Any]]) -> int:
        """Insert the chunks of a turn; returns how many were new. Idempotent on
        (session_id, turn, ord), so a redelivered message inserts nothing twice."""
        if not rows:
            return 0
        stmt = (
            insert(Chunk)
            .values(list(rows))
            .on_conflict_do_nothing(constraint="ux_chunks_session_turn_ord")
            .returning(Chunk.id)
        )
        return len((await self._session.execute(stmt)).scalars().all())

    async def delete_for_turn(self, turn_id: int) -> None:
        """Before reindexing a turn with another chunker version or embedding model."""
        await self._session.execute(delete(Chunk).where(Chunk.turn_id == turn_id))

    async def set_context(self, chunk_id: int, context: str, terms_ctx: list[str],
                          embedding_ctx: list[float], context_model: str) -> None:
        """V3+: store the LLM-generated context and the representations computed from it."""
        stmt = (
            update(Chunk)
            .where(Chunk.id == chunk_id)
            .values(context=context, terms_ctx=terms_ctx, embedding_ctx=embedding_ctx,
                    context_model=context_model)
        )
        await self._session.execute(stmt)

    # -- reads (router: R2, R3) --------------------------------------------

    async def list_candidates(
        self,
        session_id: uuid.UUID,
        before_turn: int,
        query_embedding: Sequence[float] | None,
        contextual: bool = False,
    ) -> list[ChunkCandidate]:
        """R2: every chunk of the session older than `before_turn`, with its cosine distance to
        the query computed in Postgres. Dense and BM25 ranking, fusion and dedup happen in Python
        on this one result. `contextual` reads embedding_ctx / terms_ctx and skips chunks that
        have not been contextualized yet."""
        vector_col = Chunk.embedding_ctx if contextual else Chunk.embedding
        terms_col = Chunk.terms_ctx if contextual else Chunk.terms
        distance = (
            vector_col.cosine_distance(query_embedding) if query_embedding is not None else literal(None)
        )
        stmt = (
            select(Chunk.id, Chunk.turn, Chunk.ord, Chunk.kind, Chunk.tokens, Chunk.content_hash,
                   terms_col, distance)
            .where(Chunk.session_id == session_id, Chunk.turn < before_turn)
            .order_by(Chunk.turn, Chunk.ord)
        )
        if contextual:
            stmt = stmt.where(Chunk.embedding_ctx.is_not(None))
        rows = (await self._session.execute(stmt)).all()
        return [
            ChunkCandidate(id=r[0], turn=r[1], ord=r[2], kind=r[3], tokens=r[4], content_hash=r[5],
                           terms=list(r[6] or []), distance=None if r[7] is None else float(r[7]))
            for r in rows
        ]

    async def get_contents(self, chunk_ids: Sequence[int]) -> dict[int, Chunk]:
        """R3: the full rows of the chunks that made it into the context, by id."""
        if not chunk_ids:
            return {}
        stmt = select(Chunk).where(Chunk.id.in_(list(chunk_ids)))
        return {c.id: c for c in (await self._session.execute(stmt)).scalars().all()}

    async def list_for_turn(self, turn_id: int) -> list[Chunk]:
        stmt = select(Chunk).where(Chunk.turn_id == turn_id).order_by(Chunk.ord)
        return list((await self._session.execute(stmt)).scalars().all())
