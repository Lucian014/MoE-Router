from __future__ import annotations

import uuid
from typing import Any

from sqlalchemy import exists, func, or_, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.chunk.schema import Chunk
from database.models.turn.schema import Turn


class TurnAlreadyExistsError(Exception):
    """Two prompts of the same session got the same turn number (sent in parallel)."""

    def __init__(self, session_id: uuid.UUID, turn: int):
        super().__init__(f"session {session_id} already has turn {turn}")
        self.session_id, self.turn = session_id, turn


class TurnRepository:
    """Data access for `turns`. Only flushes; get_session() owns the transaction."""

    def __init__(self, session: AsyncSession):
        self._session = session

    # -- reads -----------------------------------------------------------

    async def get(self, turn_id: int) -> Turn | None:
        return await self._session.get(Turn, turn_id)

    async def list_recent(self, session_id: uuid.UUID, limit: int) -> list[Turn]:
        """The last `limit` turns of the session, oldest first (R1). Served by ux_turns_session_turn."""
        stmt = select(Turn).where(Turn.session_id == session_id).order_by(Turn.turn.desc()).limit(limit)
        rows = list((await self._session.execute(stmt)).scalars().all())
        rows.reverse()
        return rows

    async def last_turn_number(self, session_id: uuid.UUID) -> int:
        """0 for a new or unknown session, so the next turn is always this + 1."""
        stmt = select(func.coalesce(func.max(Turn.turn), 0)).where(Turn.session_id == session_id)
        return (await self._session.execute(stmt)).scalar_one()

    async def list_to_index(self, chunker_version: str, embedding_model: str, limit: int = 100) -> list[Turn]:
        """Turns the memory service still has to (re)index: no chunks yet, or chunks produced by
        another chunker version or embedding model. Used by the reindex command."""
        any_chunk = select(Chunk.id).where(Chunk.turn_id == Turn.id)
        stale_chunk = any_chunk.where(
            or_(Chunk.chunker_version != chunker_version, Chunk.embedding_model != embedding_model)
        )
        stmt = (
            select(Turn)
            .where(or_(~exists(any_chunk), exists(stale_chunk)))
            .order_by(Turn.id)
            .limit(limit)
        )
        return list((await self._session.execute(stmt)).scalars().all())

    # -- writes ------------------------------------------------------------

    async def create(self, session_id: uuid.UUID, turn: int, user_content: str, assistant_content: str,
                     **fields: Any) -> int:
        """Insert a turn and return its id. `fields` are the optional columns of Turn (routing,
        context diagnostics, token counts). Raises TurnAlreadyExistsError on a duplicate
        (session_id, turn) instead of failing the whole transaction."""
        unknown = set(fields) - set(Turn.__table__.columns.keys())
        if unknown:
            raise ValueError(f"unknown Turn column(s): {sorted(unknown)}")
        stmt = (
            insert(Turn)
            .values(session_id=session_id, turn=turn, user_content=user_content,
                    assistant_content=assistant_content, **fields)
            .on_conflict_do_nothing(constraint="ux_turns_session_turn")
            .returning(Turn.id)
        )
        turn_id = (await self._session.execute(stmt)).scalar_one_or_none()
        if turn_id is None:
            raise TurnAlreadyExistsError(session_id, turn)
        return turn_id

    async def set_header(self, turn_id: int, header: str) -> None:
        await self._session.execute(update(Turn).where(Turn.id == turn_id).values(header=header))
