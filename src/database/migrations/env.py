import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import create_engine, pool

# Make the `database` package importable when Alembic is invoked from
# database/. parents[0]=migrations/, parents[1]=database/, parents[2]=the
# directory that CONTAINS database/ (e.g. src/) — that's what needs to be
# on sys.path for `import database...` to resolve.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from shared.config import get_postgres_settings  # noqa: E402
from database.base import Base  # noqa: E402

# Importing database.models pulls in every table's schema.py (see
# database/models/__init__.py), which is what registers each table onto
# Base.metadata. Without this import, autogenerate would see an empty
# schema and would try to DROP every real table.
import database.models  # noqa: E402,F401

config = context.config

# The URL comes from shared.config instead of alembic.ini, so migrations honor the
# same POSTGRES_* env vars as the running application. It is passed straight to
# SQLAlchemy, not through config.set_main_option(): Alembic's config is configparser,
# which chokes on a '%' in the password.

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Autogenerate support: Alembic diffs against this metadata, which now
# includes every table from every database/models/<table_name>/ folder.
target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection (`alembic upgrade --sql`)."""
    url = get_postgres_settings().sync_url.render_as_string(hide_password=False)
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
    connectable = create_engine(get_postgres_settings().sync_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
