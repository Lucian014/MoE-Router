"""models: one category + one complexity tier per model

Revision ID: b3d9e1f04a21
Revises: 635f1c5b3c2d
Create Date: 2026-10-12

Laya router redesign of the `models` table:
  + category         text NULL       one domain label per model (NULL = no category, e.g. default)
  + complexity_tier  'low' | 'high'  the level the model serves inside its category
  - task_types       text[]          replaced by category (the first element is kept)
  - description, examples            legacy of the old embedding router
  - replicas                         workers are scaled by KEDA; build_manifests uses WORKER_REPLICAS
  + ux_models_category_tier   UNIQUE (category, complexity_tier) WHERE enabled AND category IS NOT NULL
  + ux_models_single_default  UNIQUE (is_default) WHERE is_default

Existing rows get complexity_tier = 'high'. Rows that would collide on (category, tier) keep their
category only for the lowest id; the others get category NULL. Re-run
    python scripts/seed_models.py --from-yaml src/catalog/models.yaml
right after this migration to put the real categories / tiers in place.
"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "b3d9e1f04a21"
down_revision: Union[str, None] = "635f1c5b3c2d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. new columns
    op.add_column("models", sa.Column("category", sa.String(length=64), nullable=True))
    op.add_column(
        "models",
        sa.Column("complexity_tier", sa.String(length=16), nullable=False, server_default="high"),
    )
    op.create_check_constraint(
        "ck_models_complexity_tier", "models", "complexity_tier IN ('low', 'high')"
    )

    # 2. data: task_types[1] -> category (Postgres arrays are 1-based; empty array -> NULL)
    op.execute("UPDATE models SET category = task_types[1]")
    # collisions on (category, tier) among enabled rows: keep the lowest id
    op.execute("""
        UPDATE models m SET category = NULL
        WHERE m.enabled AND m.category IS NOT NULL AND EXISTS (
            SELECT 1 FROM models o
            WHERE o.enabled AND o.category = m.category
              AND o.complexity_tier = m.complexity_tier AND o.id < m.id)
    """)
    # more than one default: keep the lowest id
    op.execute("""
        UPDATE models SET is_default = false
        WHERE is_default AND id <> (SELECT min(id) FROM models WHERE is_default)
    """)

    # 3. constraints
    op.create_index(
        "ux_models_category_tier", "models", ["category", "complexity_tier"], unique=True,
        postgresql_where=sa.text("enabled AND category IS NOT NULL"),
    )
    op.execute("DROP INDEX IF EXISTS ux_models_single_default")
    op.create_index(
        "ux_models_single_default", "models", ["is_default"], unique=True,
        postgresql_where=sa.text("is_default"),
    )

    # 4. drop the old columns
    op.drop_column("models", "task_types")
    op.drop_column("models", "description")
    op.drop_column("models", "examples")
    op.drop_column("models", "replicas")


def downgrade() -> None:
    op.add_column("models", sa.Column("replicas", sa.Integer(), nullable=False, server_default="1"))
    op.add_column("models", sa.Column("examples", postgresql.ARRAY(sa.Text()), nullable=False,
                                      server_default="{}"))
    op.add_column("models", sa.Column("description", sa.Text(), nullable=False, server_default=""))
    op.add_column("models", sa.Column("task_types", postgresql.ARRAY(sa.Text()), nullable=False,
                                      server_default="{}"))
    op.execute("UPDATE models SET task_types = ARRAY[category] WHERE category IS NOT NULL")

    op.drop_index("ux_models_single_default", table_name="models")
    op.drop_index("ux_models_category_tier", table_name="models")
    op.drop_constraint("ck_models_complexity_tier", "models", type_="check")
    op.drop_column("models", "complexity_tier")
    op.drop_column("models", "category")