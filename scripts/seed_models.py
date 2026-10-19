#!/usr/bin/env python3
"""
seed_models.py -- populate the `models` table.

1. Bulk load from a YAML catalog (the human-editable source of truth):

    python scripts/seed_models.py --from-yaml src/catalog/models.yaml
    python scripts/seed_models.py --from-yaml src/catalog/models.yaml --check   # validate only

   The catalog is validated with the router's own rule (router/routing_rule.py) BEFORE touching
   the DB: exactly one enabled default, one category per model (a domain label), complexity_tier
   in {low, high}, at most one enabled model per category + tier. upsert() makes re-runs safe.
   Models that are in the DB but no longer in the YAML are left untouched (disable them with
   `enabled: false` instead of deleting the line) -- unless --disable-missing is given, which
   makes the YAML the complete catalog (the Helm setup Job uses it: a model removed from the
   chart's values has no worker any more, so the router must stop routing to it).

2. Add / update a single model:

    python scripts/seed_models.py --add --model-id "qwen2.5-coder:3b" --engine ollama \
        --category coding --tier high

Run from anywhere -- sys.path is adjusted below to find `database` and `router`.
"""
import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from database import ModelRepository, get_session
from router.domain_schema import LABELS as CATEGORIES  
from router.routing_rule import catalog_from_rows, select_model, validate_catalog  # noqa: E402


def check_catalog(models: list[dict], source: str) -> None:
    try:
        catalog = catalog_from_rows(models)
    except ValueError as e:
        raise SystemExit(f"[!] {source}: {e}")
    errors, warnings = validate_catalog(catalog, CATEGORIES)
    for w in warnings:
        print(f"[warn] {w}")
    if errors:
        raise SystemExit(f"[!] {source} is not a valid catalog:\n  " + "\n  ".join(errors))
    print(f"[OK] {source}: {len(models)} model(s)")
    for c in CATEGORIES:
        low, _ = select_model(catalog, c, "low")
        high, _ = select_model(catalog, c, "high")
        print(f"     {c:17s} low -> {low:32s} high -> {high}")


async def seed_from_yaml(yaml_path: Path, check_only: bool, disable_missing: bool = False) -> None:
    import yaml

    with open(yaml_path, encoding="utf-8") as f:
        models = (yaml.safe_load(f) or {}).get("models", [])
    if not models:
        raise SystemExit(f"[!] {yaml_path} has no models under the 'models' key.")
    for m in models:
        m.setdefault("complexity_tier", "high")
        m.setdefault("category", None)
    check_catalog(models, str(yaml_path))
    if check_only:
        return

    async with get_session() as session:      # one transaction: all or nothing
        repo = ModelRepository(session)
        # 1. free every slot the YAML touches, so swapping two models' category / tier works
        for entry in models:
            if await repo.get_by_id(entry["model_id"]) is not None:
                await repo.update(entry["model_id"], enabled=False)
        # 2. write the YAML; the default last, so _clear_other_defaults() leaves exactly one
        for entry in sorted(models, key=lambda e: e.get("is_default", False)):
            await repo.upsert(
                model_id=entry["model_id"],
                engine=entry["engine"],
                category=entry["category"],
                complexity_tier=entry["complexity_tier"],
                enabled=entry.get("enabled", True),
                is_default=entry.get("is_default", False),
            )
            print(f"[OK] upserted {entry['model_id']}")
        # 3. optionally: the YAML is the whole catalog, disable everything else
        if disable_missing:
            listed = {entry["model_id"] for entry in models}
            for row in await repo.list_enabled():
                if row.model_id not in listed:
                    await repo.update(row.model_id, enabled=False, is_default=False)
                    print(f"[OK] disabled {row.model_id} (not in {yaml_path})")
    print(f"[OK] seeded {len(models)} model(s) from {yaml_path}")


async def add_single(args) -> None:
    async with get_session() as session:
        entry = await ModelRepository(session).upsert(
            model_id=args.model_id,
            engine=args.engine,
            category=args.category,
            complexity_tier=args.tier,
            enabled=not args.disabled,
            is_default=args.default,
        )
        print(f"[OK] upserted: {entry!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-yaml", type=Path, help="models.yaml to bulk-load")
    mode.add_argument("--add", action="store_true", help="add/update a single model from CLI args")
    parser.add_argument("--check", action="store_true", help="with --from-yaml: validate only, no DB writes")
    parser.add_argument("--disable-missing", action="store_true",
                        help="with --from-yaml: disable enabled DB models that are not in the YAML")
    parser.add_argument("--model-id")
    parser.add_argument("--engine")
    parser.add_argument("--category", choices=CATEGORIES, default=None,
                        help="the one category this model serves (omit only for a generalist default)")
    parser.add_argument("--tier", choices=["low", "high"], default="high")
    parser.add_argument("--disabled", action="store_true")
    parser.add_argument("--default", action="store_true", help="make this the fallback model")
    args = parser.parse_args()

    if args.from_yaml:
        asyncio.run(seed_from_yaml(args.from_yaml, args.check, args.disable_missing))
    else:
        if not args.model_id or not args.engine:
            parser.error("--add requires --model-id and --engine")
        if args.category is None and not args.default:
            print("[warn] no --category and not --default: this model will never be selected")
        asyncio.run(add_single(args))


if __name__ == "__main__":
    main()