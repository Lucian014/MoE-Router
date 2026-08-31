import os


POSTGRES_HOST = os.getenv("POSTGRES_HOST", "postgres.platform.svc.cluster.local")
POSTGRES_PORT = os.getenv("POSTGRES_PORT", "5432")
POSTGRES_DB = os.getenv("POSTGRES_DB", "model_registry")
POSTGRES_USER = os.getenv("POSTGRES_USER")
POSTGRES_PASSWORD = os.getenv("POSTGRES_PASSWORD")

# Async URL (asyncpg) — used by the application at runtime.
ASYNC_DATABASE_URL = os.getenv(
    "ASYNC_DATABASE_URL",
    f"postgresql+asyncpg://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)

# Sync URL (psycopg2) — used by Alembic, which does not support async
SYNC_DATABASE_URL = os.getenv(
    "SYNC_DATABASE_URL",
    f"postgresql+psycopg2://{POSTGRES_USER}:{POSTGRES_PASSWORD}"
    f"@{POSTGRES_HOST}:{POSTGRES_PORT}/{POSTGRES_DB}",
)
