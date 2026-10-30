"""session memory: sessions, turns, chunks + context limits on models

Revision ID: c4f8e2a7d913
Revises: b3d9e1f04a21
Create Date: 2026-10-30

Shared conversation context for every model (docs/rag-planv2.md):
  + extension vector   needs the pgvector/pgvector:pg16 image (same major version as postgres:16)
  + sessions           one row per conversation, with the effective routing state
  + turns              one row per prompt + answer, written by the worker before it replies
  + chunks             retrieval units of a turn, written by the memory service
  + models.context_window     effective context at run time (Ollama num_ctx / vLLM --max-model-len),
                              input + output together
  + models.max_output_tokens  reserved for the answer

Written by hand: autogenerate does not emit CREATE EXTENSION and gets vector(1024) wrong.
The unique constraints on turns (session_id, turn) and chunks (session_id, turn, ord) also serve
the "last N turns" and "chunks of the session" lookups, so those need no separate index.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "c4f8e2a7d913"
down_revision: Union[str, None] = "b3d9e1f04a21"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

EMBEDDING_DIM = 1024  # bge-m3


def _timestamps(*names: str) -> list[sa.Column]:
    return [sa.Column(n, sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now())
            for n in names]


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 1. context limits per model; existing rows get the defaults
    op.add_column("models", sa.Column("context_window", sa.Integer(), nullable=False, server_default="4096"))
    op.add_column("models", sa.Column("max_output_tokens", sa.Integer(), nullable=False, server_default="512"))
    op.create_check_constraint(
        "ck_models_context_limits", "models",
        "max_output_tokens > 0 AND context_window > max_output_tokens",
    )

    # 2. sessions
    op.create_table(
        "sessions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("user_id", sa.Text(), nullable=True),
        sa.Column("domain", sa.String(length=64), nullable=True),
        sa.Column("domain_confidence", sa.REAL(), nullable=True),
        sa.Column("complexity_score", sa.REAL(), nullable=True),
        *_timestamps("created_at", "last_activity_at"),
    )

    # 3. turns
    op.create_table(
        "turns",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("sessions.id", ondelete="CASCADE", name="fk_turns_session"), nullable=False),
        sa.Column("turn", sa.Integer(), nullable=False),
        sa.Column("user_content", sa.Text(), nullable=False),
        sa.Column("assistant_content", sa.Text(), nullable=False),
        sa.Column("header", sa.Text(), nullable=True),
        sa.Column("model_id", sa.String(length=255), nullable=True),
        sa.Column("predicted_domain", sa.String(length=64), nullable=True),
        sa.Column("predicted_complexity", sa.REAL(), nullable=True),
        sa.Column("effective_domain", sa.String(length=64), nullable=True),
        sa.Column("effective_complexity", sa.REAL(), nullable=True),
        sa.Column("inherited", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("rag_variant", sa.String(length=32), nullable=True),
        sa.Column("recent_turns", sa.Integer(), nullable=True),
        sa.Column("retrieved_chunk_ids", postgresql.ARRAY(sa.BigInteger()), nullable=True),
        sa.Column("prompt_tokens", sa.Integer(), nullable=True),
        sa.Column("completion_tokens", sa.Integer(), nullable=True),
        *_timestamps("created_at"),
        sa.UniqueConstraint("session_id", "turn", name="ux_turns_session_turn"),
        sa.CheckConstraint("turn >= 1", name="ck_turns_turn_positive"),
    )

    # 4. chunks
    op.create_table(
        "chunks",
        sa.Column("id", sa.BigInteger(), sa.Identity(), primary_key=True),
        sa.Column("turn_id", sa.BigInteger(),
                  sa.ForeignKey("turns.id", ondelete="CASCADE", name="fk_chunks_turn"), nullable=False),
        sa.Column("session_id", postgresql.UUID(as_uuid=True),
                  sa.ForeignKey("sessions.id", ondelete="CASCADE", name="fk_chunks_session"), nullable=False),
        sa.Column("turn", sa.Integer(), nullable=False),
        sa.Column("ord", sa.Integer(), nullable=False),
        sa.Column("source", sa.String(length=16), nullable=False),
        sa.Column("kind", sa.String(length=16), nullable=False),
        sa.Column("lang", sa.String(length=32), nullable=True),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("tokens", sa.Integer(), nullable=False),
        sa.Column("terms", postgresql.ARRAY(sa.Text()), nullable=False),
        sa.Column("embedding", Vector(EMBEDDING_DIM), nullable=False),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("terms_ctx", postgresql.ARRAY(sa.Text()), nullable=True),
        sa.Column("embedding_ctx", Vector(EMBEDDING_DIM), nullable=True),
        sa.Column("embedding_model", sa.String(length=128), nullable=False),
        sa.Column("context_model", sa.String(length=128), nullable=True),
        sa.Column("chunker_version", sa.String(length=32), nullable=False),
        *_timestamps("created_at"),
        sa.UniqueConstraint("session_id", "turn", "ord", name="ux_chunks_session_turn_ord"),
        sa.CheckConstraint("source IN ('prompt', 'response')", name="ck_chunks_source"),
        sa.CheckConstraint("kind IN ('prose', 'code', 'log', 'math', 'list', 'table')", name="ck_chunks_kind"),
        sa.CheckConstraint("ord >= 0", name="ck_chunks_ord"),
        sa.CheckConstraint("tokens > 0", name="ck_chunks_tokens"),
    )
    # the foreign key alone is not indexed: needed by the ON DELETE CASCADE from turns
    # and by the reindex query ("turns without chunks")
    op.create_index("ix_chunks_turn_id", "chunks", ["turn_id"])


def downgrade() -> None:
    op.drop_index("ix_chunks_turn_id", table_name="chunks")
    op.drop_table("chunks")
    op.drop_table("turns")
    op.drop_table("sessions")
    op.drop_constraint("ck_models_context_limits", "models", type_="check")
    op.drop_column("models", "max_output_tokens")
    op.drop_column("models", "context_window")
    op.execute("DROP EXTENSION IF EXISTS vector")
