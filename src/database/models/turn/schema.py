import datetime
import uuid

from sqlalchemy import (
    REAL,
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class Turn(Base):
    """One prompt + answer. Written by the worker in the same transaction that updates the
    session, before it replies, so the next prompt always sees it (docs/rag-planv2.md, 2).

    `user_content` is the user's original prompt, never the assembled context: the context
    is rebuilt for every request and not stored. `header` is filled in later by the memory
    service when it chunks the turn."""

    __tablename__ = "turns"
    __table_args__ = (
        UniqueConstraint("session_id", "turn", name="ux_turns_session_turn"),
        CheckConstraint("turn >= 1", name="ck_turns_turn_positive"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("sessions.id", ondelete="CASCADE", name="fk_turns_session"), nullable=False
    )
    turn: Mapped[int] = mapped_column(Integer, nullable=False)

    user_content: Mapped[str] = mapped_column(Text, nullable=False)
    assistant_content: Mapped[str] = mapped_column(Text, nullable=False)
    header: Mapped[str | None] = mapped_column(Text, nullable=True)

    # routing: what the classifier predicted and what was actually used (inherited or not)
    model_id: Mapped[str | None] = mapped_column(String(255), nullable=True)
    predicted_domain: Mapped[str | None] = mapped_column(String(64), nullable=True)
    predicted_complexity: Mapped[float | None] = mapped_column(REAL, nullable=True)
    effective_domain: Mapped[str | None] = mapped_column(String(64), nullable=True)
    effective_complexity: Mapped[float | None] = mapped_column(REAL, nullable=True)
    inherited: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, server_default="false")

    # context: how it was assembled (diagnostics and evaluation)
    rag_variant: Mapped[str | None] = mapped_column(String(32), nullable=True)
    recent_turns: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retrieved_chunk_ids: Mapped[list[int] | None] = mapped_column(ARRAY(BigInteger), nullable=True)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    completion_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:
        return f"<Turn id={self.id} session_id={self.session_id} turn={self.turn} model_id={self.model_id!r}>"
