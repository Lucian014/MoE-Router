"""
Script de test manual pentru ModelRepository, rulat printr-un tunel deschis
manual (kubectl port-forward) sau direct dacă rulezi din interiorul
clusterului.

Rulare:
    export POSTGRES_HOST=localhost POSTGRES_PORT=5432 \
           POSTGRES_DB=model_registry POSTGRES_USER=... POSTGRES_PASSWORD=...
    python test_crud.py

Nu e un test pytest formal — e un smoke test rapid ca să confirmi că
sesiunea, repository-ul și commit/rollback-ul chiar funcționează cap-coadă
împotriva bazei reale (sau a unei baze de test separate, ideal).
"""

import asyncio
import sys
from pathlib import Path

# Make the `database` package importable regardless of where this script is
# invoked from. This file lives at <repo_root>/scripts/test_crud.py, and
# `database` lives at <repo_root>/src/database — so parents[1] is
# <repo_root>, and we need <repo_root>/src on sys.path. Mirrors the same
# approach used in migrations/env.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from database import ModelAlreadyExistsError, ModelNotFoundError, ModelRepository, get_session  # noqa: E402


async def main() -> None:
    test_id = "test-model:smoke-check"

    # -- CLEANUP FROM ANY PREVIOUS FAILED RUN --------------------------------
    # If a prior run crashed partway through (e.g. an assertion failed before
    # reaching the DELETE step), the row is left behind and the next run's
    # CREATE fails with ModelAlreadyExistsError. Make the script idempotent
    # by clearing any leftover row before starting.
    async with get_session() as session:
        repo = ModelRepository(session)
        existing = await repo.get_by_id(test_id)
        if existing is not None:
            print(f"[SETUP]  found leftover row from a previous run, deleting it first")
            await repo.delete(test_id)

    try:
        await _run_checks(test_id)
    finally:
        # Guarantee cleanup even if an assertion fails midway, so the next
        # run starts clean without needing the block above.
        async with get_session() as session:
            repo = ModelRepository(session)
            if await repo.get_by_id(test_id) is not None:
                await repo.delete(test_id)
                print(f"[CLEANUP] removed {test_id} (finally block)")


async def _run_checks(test_id: str) -> None:
    async with get_session() as session:
        repo = ModelRepository(session)
        entry = await repo.create(
            model_id=test_id,
            engine="vllm",
            description="Smoke test entry",
            examples=["hello", "world"],
            replicas=2,
            enabled=True,
        )
        print(f"[CREATE] {entry!r}")

    # -- READ ----------------------------------------------------------------
    async with get_session() as session:
        repo = ModelRepository(session)
        fetched = await repo.get_by_id_or_raise(test_id)
        assert fetched.engine == "vllm"
        assert fetched.replicas == 2
        print(f"[READ]   {fetched!r}")

        all_entries = await repo.list_all()
        assert any(e.model_id == test_id for e in all_entries)
        print(f"[READ]   list_all() has {len(all_entries)} entries total")

    # -- DUPLICATE CREATE SHOULD FAIL ---------------------------------------
    async with get_session() as session:
        repo = ModelRepository(session)
        try:
            await repo.create(model_id=test_id, engine="vllm")
            raise SystemExit("[FAIL] expected ModelAlreadyExistsError, none raised")
        except ModelAlreadyExistsError:
            print("[OK]     duplicate create correctly rejected")

    # -- UPDATE --------------------------------------------------------------
    async with get_session() as session:
        repo = ModelRepository(session)
        updated = await repo.update(test_id, replicas=5, enabled=False)
        assert updated.replicas == 5
        assert updated.enabled is False
        print(f"[UPDATE] {updated!r}")

    # confirm the update was actually committed (new session, fresh read)
    async with get_session() as session:
        repo = ModelRepository(session)
        reread = await repo.get_by_id_or_raise(test_id)
        assert reread.replicas == 5
        assert reread.enabled is False
        print(f"[VERIFY] update persisted across sessions: {reread!r}")

    # -- ROLLBACK BEHAVIOR (exception must propagate OUT of the `async with`
    # block for get_session() to see it and roll back. Catching it inside
    # the block, like `except RuntimeError: pass`, hides it from the context
    # manager and the change gets committed instead — that was a bug in an
    # earlier version of this script, not in get_session() itself.) --------
    try:
        async with get_session() as session:
            repo = ModelRepository(session)
            await repo.update(test_id, replicas=999)
            raise RuntimeError("simulated failure after staging a change")
    except RuntimeError:
        pass  # expected: get_session() rolled back before re-raising this

    async with get_session() as session:
        repo = ModelRepository(session)
        after_failed_txn = await repo.get_by_id_or_raise(test_id)
        assert after_failed_txn.replicas == 5, (
            f"expected rollback to keep replicas=5, got {after_failed_txn.replicas}"
        )
        print("[OK]     failed transaction correctly rolled back (replicas still 5)")

    # -- DELETE ----------------------------------------------------------------
    async with get_session() as session:
        repo = ModelRepository(session)
        await repo.delete(test_id)
        print(f"[DELETE] {test_id} removed")

    async with get_session() as session:
        repo = ModelRepository(session)
        try:
            await repo.get_by_id_or_raise(test_id)
            raise SystemExit("[FAIL] entry still exists after delete")
        except ModelNotFoundError:
            print("[OK]     confirmed entry no longer exists")

    print("\nAll CRUD + rollback checks passed.")


if __name__ == "__main__":
    asyncio.run(main())