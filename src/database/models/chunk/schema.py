import datetime
import uuid

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Identity,
    Index,
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

EMBEDDING_DIM = 1024  # bge-m3
CHUNK_SOURCES = ("prompt", "response")
CHUNK_KINDS = ("prose", "code", "log", "math", "list", "table")


class Chunk(Base):
    """A retrieval unit of a turn, written by the memory service (docs/rag-planv2.md, 3 and 4.3).

    Every representation lives on the same row, so all RAG variants are evaluated on the same
    chunks: `embedding` / `terms` (V1, V2) are always set, `context` / `terms_ctx` /
    `embedding_ctx` (V3+) stay NULL until a chunk is contextualized. `content` never includes
    the turn header; the header only goes into the text that `embedding` and `terms` are
    computed from."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("session_id", "turn", "ord", name="ux_chunks_session_turn_ord"),
        CheckConstraint("source IN ('prompt', 'response')", name="ck_chunks_source"),
        CheckConstraint("kind IN ('prose', 'code', 'log', 'math', 'list', 'table')", name="ck_chunks_kind"),
        CheckConstraint("ord >= 0", name="ck_chunks_ord"),
        CheckConstraint("tokens > 0", name="ck_chunks_tokens"),
        Index("ix_chunks_turn_id", "turn_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    turn_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("turns.id", ondelete="CASCADE", name="fk_chunks_turn"), nullable=False
    )
    session_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("sessions.id", ondelete="CASCADE", name="fk_chunks_session"), nullable=False
    )
    turn: Mapped[int] = mapped_column(Integer, nullable=False)
    ord: Mapped[int] = mapped_column(Integer, nullable=False)

    source: Mapped[str] = mapped_column(String(16), nullable=False)
    kind: Mapped[str] = mapped_column(String(16), nullable=False)
    lang: Mapped[str | None] = mapped_column(String(32), nullable=True)

    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    tokens: Mapped[int] = mapped_column(Integer, nullable=False)

    # V1 / V2: from "Întrebare: {header} | {content}"
    terms: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)

    # V3+: from "{context}\n{content}", NULL until contextualized
    context: Mapped[str | None] = mapped_column(Text, nullable=True)
    terms_ctx: Mapped[list[str] | None] = mapped_column(ARRAY(Text), nullable=True)
    embedding_ctx: Mapped[list[float] | None] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)

    # provenance: rows produced by another model or chunker version are not mixed
    embedding_model: Mapped[str] = mapped_column(String(128), nullable=False)
    context_model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    chunker_version: Mapped[str] = mapped_column(String(32), nullable=False)

    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    def __repr__(self) -> str:
        return (
            f"<Chunk id={self.id} session_id={self.session_id} turn={self.turn} ord={self.ord} "
            f"source={self.source!r} kind={self.kind!r} tokens={self.tokens}>"
        )
