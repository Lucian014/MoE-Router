from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from database.models.model_registry.schema import (
    COMPLEXITY_TIERS,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_MAX_OUTPUT_TOKENS,
    ModelRegistryEntry,
)


class ModelNotFoundError(Exception):
    def __init__(self, model_id: str):
        super().__init__(f"Model '{model_id}' not found in registry.")
        self.model_id = model_id


class ModelAlreadyExistsError(Exception):
    def __init__(self, model_id: str):
        super().__init__(f"Model '{model_id}' already exists in registry.")
        self.model_id = model_id


class SlotTakenError(Exception):
    """Another ENABLED model already serves this (category, complexity_tier)."""

    def __init__(self, category: str, tier: str, other_model_id: str):
        super().__init__(
            f"category {category!r} / tier {tier!r} is already served by enabled model "
            f"'{other_model_id}'. Disable it first (or give this model another category / tier)."
        )
        self.category, self.tier, self.other_model_id = category, tier, other_model_id


def _check_tier(tier: str) -> None:
    if tier not in COMPLEXITY_TIERS:
        raise ValueError(f"complexity_tier must be one of {COMPLEXITY_TIERS}, got {tier!r}")


def _check_context_limits(context_window: int, max_output_tokens: int) -> None:
    """Mirrors ck_models_context_limits: the answer must leave room for the prompt."""
    if max_output_tokens <= 0 or context_window <= max_output_tokens:
        raise ValueError(
            f"need 0 < max_output_tokens < context_window, got max_output_tokens={max_output_tokens}, "
            f"context_window={context_window}"
        )


class ModelRepository:
    """
    Thin data-access layer around ModelRegistryEntry.

    Takes an AsyncSession rather than owning one, so callers control the transaction boundary
    (see database.session.get_session). Methods only stage changes; the surrounding
    `get_session()` context manager commits or rolls back.

    The DB itself enforces "one enabled model per (category, tier)" and "one default" with
    partial unique indexes; the checks below only turn those into readable errors.
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

    async def list_for_category(self, category: str) -> list[ModelRegistryEntry]:
        """Enabled models of one category (0, 1 or 2 rows: low and/or high)."""
        stmt = (
            select(ModelRegistryEntry)
            .where(ModelRegistryEntry.enabled.is_(True))
            .where(ModelRegistryEntry.category == category)
            .order_by(ModelRegistryEntry.complexity_tier)
        )
        result = await self._session.execute(stmt)
        return list(result.scalars().all())

    async def get_default(self) -> ModelRegistryEntry | None:
        stmt = select(ModelRegistryEntry).where(ModelRegistryEntry.is_default.is_(True))
        result = await self._session.execute(stmt)
        return result.scalar_one_or_none()

    # -- writes ------------------------------------------------------------

    async def create(
        self,
        model_id: str,
        engine: str,
        category: str | None = None,
        complexity_tier: str = "high",
        enabled: bool = True,
        is_default: bool = False,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> ModelRegistryEntry:
        if await self.get_by_id(model_id) is not None:
            raise ModelAlreadyExistsError(model_id)
        _check_tier(complexity_tier)
        _check_context_limits(context_window, max_output_tokens)
        if enabled:
            await self._check_slot_free(category, complexity_tier, exclude_model_id=model_id)
        if is_default:
            await self._clear_other_defaults(exclude_model_id=model_id)

        entry = ModelRegistryEntry(
            model_id=model_id,
            engine=engine,
            category=category,
            complexity_tier=complexity_tier,
            enabled=enabled,
            is_default=is_default,
            context_window=context_window,
            max_output_tokens=max_output_tokens,
        )
        self._session.add(entry)
        await self._session.flush()
        return entry

    async def upsert(
        self,
        model_id: str,
        engine: str,
        category: str | None = None,
        complexity_tier: str = "high",
        enabled: bool = True,
        is_default: bool = False,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    ) -> ModelRegistryEntry:
        """Create the model if it doesn't exist, otherwise update it in place
        (safe to re-run seed_models.py after editing models.yaml)."""
        entry = await self.get_by_id(model_id)
        if entry is None:
            return await self.create(
                model_id=model_id,
                engine=engine,
                category=category,
                complexity_tier=complexity_tier,
                enabled=enabled,
                is_default=is_default,
                context_window=context_window,
                max_output_tokens=max_output_tokens,
            )

        _check_tier(complexity_tier)
        _check_context_limits(context_window, max_output_tokens)
        if enabled:
            await self._check_slot_free(category, complexity_tier, exclude_model_id=model_id)
        if is_default:
            await self._clear_other_defaults(exclude_model_id=model_id)

        entry.engine = engine
        entry.category = category
        entry.complexity_tier = complexity_tier
        entry.enabled = enabled
        entry.is_default = is_default
        entry.context_window = context_window
        entry.max_output_tokens = max_output_tokens
        await self._session.flush()
        return entry

    async def update(
        self,
        model_id: str,
        *,
        engine: str | None = None,
        category: str | None = None,
        clear_category: bool = False,
        complexity_tier: str | None = None,
        enabled: bool | None = None,
        is_default: bool | None = None,
        context_window: int | None = None,
        max_output_tokens: int | None = None,
    ) -> ModelRegistryEntry:
        """Partial update. `category=None` means "leave unchanged"; use clear_category=True to set
        it to NULL."""
        entry = await self.get_by_id_or_raise(model_id)

        new_category = None if clear_category else (category if category is not None else entry.category)
        new_tier = complexity_tier if complexity_tier is not None else entry.complexity_tier
        new_enabled = enabled if enabled is not None else entry.enabled
        new_context = context_window if context_window is not None else entry.context_window
        new_max_output = max_output_tokens if max_output_tokens is not None else entry.max_output_tokens
        _check_tier(new_tier)
        _check_context_limits(new_context, new_max_output)
        if new_enabled:
            await self._check_slot_free(new_category, new_tier, exclude_model_id=model_id)

        if engine is not None:
            entry.engine = engine
        entry.category = new_category
        entry.complexity_tier = new_tier
        entry.enabled = new_enabled
        entry.context_window = new_context
        entry.max_output_tokens = new_max_output
        if is_default is not None:
            if is_default:
                await self._clear_other_defaults(exclude_model_id=model_id)
            entry.is_default = is_default
        await self._session.flush()
        return entry

    async def set_enabled(self, model_id: str, enabled: bool) -> ModelRegistryEntry:
        return await self.update(model_id, enabled=enabled)

    async def set_category(self, model_id: str, category: str | None) -> ModelRegistryEntry:
        if category is None:
            return await self.update(model_id, clear_category=True)
        return await self.update(model_id, category=category)

    async def set_complexity_tier(self, model_id: str, tier: str) -> ModelRegistryEntry:
        return await self.update(model_id, complexity_tier=tier)

    async def set_default(self, model_id: str) -> ModelRegistryEntry:
        return await self.update(model_id, is_default=True)

    async def delete(self, model_id: str) -> None:
        entry = await self.get_by_id_or_raise(model_id)
        await self._session.delete(entry)
        await self._session.flush()

    # -- helpers -----------------------------------------------------------

    async def _check_slot_free(self, category: str | None, tier: str, exclude_model_id: str) -> None:
        if category is None:
            return
        stmt = select(ModelRegistryEntry).where(
            ModelRegistryEntry.enabled.is_(True),
            ModelRegistryEntry.category == category,
            ModelRegistryEntry.complexity_tier == tier,
            ModelRegistryEntry.model_id != exclude_model_id,
        )
        other = (await self._session.execute(stmt)).scalars().first()
        if other is not None:
            raise SlotTakenError(category, tier, other.model_id)

    async def _clear_other_defaults(self, exclude_model_id: str) -> None:
        stmt = select(ModelRegistryEntry).where(
            ModelRegistryEntry.is_default.is_(True),
            ModelRegistryEntry.model_id != exclude_model_id,
        )
        result = await self._session.execute(stmt)
        for entry in result.scalars().all():
            entry.is_default = False
        await self._session.flush()