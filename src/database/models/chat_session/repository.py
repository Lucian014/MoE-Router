from __future__ import annotations

import uuid

from sqlalchemy import delete, func, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.chat_session.schema import ChatSession


class ChatSessionRepository:
    """Data access for `sessions`. Like every repository here it only flushes; the surrounding
    get_session() owns the transaction."""

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get(self, session_id: uuid.UUID) -> ChatSession | None:
        return await self._session.get(ChatSession, session_id)

    async def ensure(self, session_id: uuid.UUID, user_id: str | None = None) -> None:
        """Create the session if it does not exist yet. The gateway hands out the id, so the first
        turn of a conversation is also the moment its row is written."""
        stmt = (
            insert(ChatSession)
            .values(id=session_id, user_id=user_id)
            .on_conflict_do_nothing(index_elements=[ChatSession.id])
        )
        await self._session.execute(stmt)

    async def update_routing_state(
        self,
        session_id: uuid.UUID,
        domain: str | None,
        domain_confidence: float | None,
        complexity_score: float | None,
    ) -> None:
        """Store the effective routing of the turn just answered and mark the session active."""
        stmt = (
            update(ChatSession)
            .where(ChatSession.id == session_id)
            .values(
                domain=domain,
                domain_confidence=domain_confidence,
                complexity_score=complexity_score,
                last_activity_at=func.now(),
            )
        )
        await self._session.execute(stmt)

    async def delete(self, session_id: uuid.UUID) -> None:
        """Deletes the session with its turns and chunks (ON DELETE CASCADE)."""
        await self._session.execute(delete(ChatSession).where(ChatSession.id == session_id))
