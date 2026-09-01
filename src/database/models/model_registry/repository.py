from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.model_registry.schema import ModelRegistryEntry


class ModelNotFoundError(Exception):
    def __init__(self, model_id: str):
        super().__init__(f"Model '{model_id}' not found in registry.")
        self.model_id = model_id


class ModelAlreadyExistsError(Exception):
    def __init__(self, model_id: str):
        super().__init__(f"Model '{model_id}' already exists in registry.")
        self.model_id = model_id


class ModelRepository:
    """
    Thin data-access layer around ModelRegistryEntry.

    Takes an AsyncSession rather than owning one, so callers control the
    transaction boundary (see database.session.get_session). Every method
    below only stages changes via `session.add` / delete / mutation; the
    surrounding `get_session()` context manager is what commits or rolls
    back.
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    # -- reads -----------------------------------------------------------

    async def get_by_id(self, model_id: str) -> ModelRegistryEntry | None:
        stmt = select(ModelRegistryEntry).where(ModelRegistryEntry.model_id == model_id)
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    async def get_by_id_or_raise(self, model_id: str) -> ModelRegistryEntry:
        entry = await self.get_by_id(model_id)
        if entry is None:
            raise ModelNotFoundError(model_id)
        return entry

    async def list_all(self) -> list[ModelRegistryEntry]:
        stmt = select(ModelRegistryEntry).order_by(ModelRegistryEntry.model_id)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_enabled(self) -> list[ModelRegistryEntry]:
        """The query the router will eventually call at startup in place
        of parsing models.json."""
        stmt = (
            select(ModelRegistryEntry)
            .where(ModelRegistryEntry.enabled.is_(True))
            .order_by(ModelRegistryEntry.model_id)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def list_by_engine(self, engine: str) -> list[ModelRegistryEntry]:
        stmt = select(ModelRegistryEntry).where(ModelRegistryEntry.engine == engine)
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    # -- writes ------------------------------------------------------------

    async def create(
        self,
        model_id: str,
        engine: str,
        description: str = "",
        examples: list[str] | None = None,
        replicas: int = 1,
        enabled: bool = True,
    ) -> ModelRegistryEntry:
        if await self.get_by_id(model_id) is not None:
            raise ModelAlreadyExistsError(model_id)

        entry = ModelRegistryEntry(
            model_id=model_id,
            engine=engine,
            description=description,
            examples=examples or [],
            replicas=replicas,
            enabled=enabled,
        )
        self._session.add(entry)
        await self._session.flush()  # populate defaults/id without committing
        return entry

    async def upsert(
        self,
        model_id: str,
        engine: str,
        description: str = "",
        examples: list[str] | None = None,
        replicas: int = 1,
        enabled: bool = True,
    ) -> ModelRegistryEntry:
        """Create the model if it doesn't exist, otherwise update it in
        place. Handy for seeding/import scripts run repeatedly (e.g. a
        future `manage_models.py` step that syncs from k8s manifests)."""
        entry = await self.get_by_id(model_id)
        if entry is None:
            return await self.create(
                model_id=model_id,
                engine=engine,
                description=description,
                examples=examples,
                replicas=replicas,
                enabled=enabled,
            )

        entry.engine = engine
        entry.description = description
        entry.examples = examples or []
        entry.replicas = replicas
        entry.enabled = enabled
        await self._session.flush()
        return entry

    async def update(
        self,
        model_id: str,
        *,
        engine: str | None = None,
        description: str | None = None,
        examples: list[str] | None = None,
        replicas: int | None = None,
        enabled: bool | None = None,
    ) -> ModelRegistryEntry:
        entry = await self.get_by_id_or_raise(model_id)

        if engine is not None:
            entry.engine = engine
        if description is not None:
            entry.description = description
        if examples is not None:
            entry.examples = examples
        if replicas is not None:
            entry.replicas = replicas
        if enabled is not None:
            entry.enabled = enabled

        await self._session.flush()
        return entry

    async def set_enabled(self, model_id: str, enabled: bool) -> ModelRegistryEntry:
        return await self.update(model_id, enabled=enabled)

    async def set_replicas(self, model_id: str, replicas: int) -> ModelRegistryEntry:
        return await self.update(model_id, replicas=replicas)

    async def delete(self, model_id: str) -> None:
        entry = await self.get_by_id_or_raise(model_id)
        await self._session.delete(entry)
        await self._session.flush()
