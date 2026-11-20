"""models.inference_host

Revision ID: f2b6d8a41c07
Revises: e1a7c3b95d20
Create Date: 2026-11-20

The generic workers serve every enabled model, so where a model runs is read from the registry
instead of a per-model Deployment's INFERENCE_HOST. NULL = the workers' default Ollama (OLLAMA_URL);
an external server (vLLM) needs its address here.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "f2b6d8a41c07"
down_revision: Union[str, None] = "e1a7c3b95d20"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("models", sa.Column("inference_host", sa.String(512), nullable=True))


def downgrade() -> None:
    op.drop_column("models", "inference_host")
