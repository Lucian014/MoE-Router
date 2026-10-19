from contextlib import asynccontextmanager
from functools import lru_cache
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from shared.config import get_postgres_settings


# The engine is created on first use, not at import: importing `database` must not
# require POSTGRES_* to be set (the router image imports it at build time).
@lru_cache
def get_engine() -> AsyncEngine:
    return create_async_engine(
        get_postgres_settings().async_url,
        pool_pre_ping=True,
        future=True,
    )


@lru_cache
def _session_factory() -> async_sessionmaker[AsyncSession]:
    return async_sessionmaker(
        bind=get_engine(),
        expire_on_commit=False,
        autoflush=False,
    )


@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """
    Usage:
        async with get_session() as session:
            repo = ModelRepository(session)
            ...

    Commits on clean exit, rolls back on exception, always closes.
    Shared by every table's repository — none of them open their own
    engine or session factory.
    """
    session = _session_factory()()
    try:
        yield session
        await session.commit()
    except Exception:
        await session.rollback()
        raise
    finally:
        await session.close()


async def dispose_engine() -> None:
    """Call on service shutdown to close the connection pool cleanly."""
    if get_engine.cache_info().currsize:
        await get_engine().dispose()
