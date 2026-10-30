import datetime
import uuid

from sqlalchemy import REAL, DateTime, String, Text, Uuid, func
from sqlalchemy.orm import Mapped, mapped_column

from database.base import Base


class ChatSession(Base):
    """One conversation. Holds the effective routing state of its last turn, which the router
    inherits when a prompt is too short or too ambiguous to classify (docs/rag-planv2.md, 4.1).

    Named ChatSession, not Session, to stay clear of SQLAlchemy's own Session."""

    __tablename__ = "sessions"

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    user_id: Mapped[str | None] = mapped_column(Text, nullable=True)

    domain: Mapped[str | None] = mapped_column(String(64), nullable=True)
    domain_confidence: Mapped[float | None] = mapped_column(REAL, nullable=True)
    complexity_score: Mapped[float | None] = mapped_column(REAL, nullable=True)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    last_activity_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<ChatSession id={self.id} domain={self.domain!r} "
            f"complexity_score={self.complexity_score}>"
        )
