from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from database.config import ASYNC_DATABASE_URL

engine = create_async_engine(
    ASYNC_DATABASE_URL,
    pool_pre_ping=True,
    future=True,
)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
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
    session = AsyncSessionLocal()
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
    await engine.dispose()
