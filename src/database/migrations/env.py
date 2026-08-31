import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make the `database` package importable when Alembic is invoked from
# database/. parents[0]=migrations/, parents[1]=database/, parents[2]=the
# directory that CONTAINS database/ (e.g. src/) — that's what needs to be
# on sys.path for `import database...` to resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from database.config import SYNC_DATABASE_URL  # noqa: E402
from database.base import Base  # noqa: E402

# Importing database.models pulls in every table's schema.py (see
# database/models/__init__.py), which is what registers each table onto
# Base.metadata. Without this import, autogenerate would see an empty
# schema and would try to DROP every real table.
import database.models  # noqa: E402,F401

config = context.config

# Inject the URL from database.config instead of alembic.ini, so migrations
# honor the same POSTGRES_* env vars as the running application.
config.set_main_option("sqlalchemy.url", SYNC_DATABASE_URL)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Autogenerate support: Alembic diffs against this metadata, which now
# includes every table from every database/models/<table_name>/ folder.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection (`alembic upgrade --sql`)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live DB connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
