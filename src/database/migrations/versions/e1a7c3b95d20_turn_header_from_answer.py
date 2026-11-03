"""turns.header_from_answer

Revision ID: e1a7c3b95d20
Revises: c4f8e2a7d913
Create Date: 2026-11-02

The header of a turn whose prompt has no prose (only code or a log) is the first sentence of the
answer (docs/rag-planv2.md 4.2). Retrieved excerpts show it as "Topic: ..." instead of
"Question: ...", so the router needs to know which kind it is. Turns indexed before this
revision keep false; `memory.service --reindex --force` recomputes it.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "e1a7c3b95d20"
down_revision: Union[str, None] = "c4f8e2a7d913"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("turns", sa.Column("header_from_answer", sa.Boolean(), nullable=False,
                                     server_default=sa.false()))


def downgrade() -> None:
    op.drop_column("turns", "header_from_answer")
