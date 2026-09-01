#!/usr/bin/env python3
"""
seed_models.py — populate the `models` table.

Two modes:

1. Bulk load from a YAML file (e.g. keep using catalog/models.yaml as a
   human-editable source of truth, and sync it into the DB):

    python scripts/seed_models.py --from-yaml catalog/models.yaml

   Uses upsert(), so re-running this after editing the YAML is safe —
   existing models are updated in place, new ones are created.

2. Add a single model from the command line:

    python scripts/seed_models.py --add \
        --model-id "qwen2.5-coder:7b" \
        --engine ollama \
        --description "Specialized coding model..." \
        --example "Write a Python function that parses a JSON file" \
        --replicas 1

Run from anywhere — sys.path is adjusted below to find the `database`
package regardless of current working directory.
"""

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from database import get_session, ModelRepository


async def seed_from_yaml(yaml_path: Path) -> None:
    import yaml

    if not yaml_path.exists():
        raise FileNotFoundError(f"{yaml_path} not found")

    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    models = data.get("models", [])
    if not models:
        raise ValueError(f"{yaml_path} has no models under the 'models' key.")

    async with get_session() as session:
        repo = ModelRepository(session)

        for entry in models:
            model_id = entry["model_id"]
            await repo.upsert(
                model_id=model_id,
                engine=entry["engine"],
                description=entry.get("description", ""),
                examples=entry.get("examples", []),
                replicas=entry.get("replicas", 1),
                enabled=entry.get("enabled", True),
            )
            print(f"[OK] Upserted '{model_id}'")

    print(f"[OK] Seeded {len(models)} model(s) from {yaml_path}")


async def add_single(
    model_id: str,
    engine: str,
    description: str,
    examples: list[str],
    replicas: int,
    enabled: bool,
) -> None:
    async with get_session() as session:
        repo = ModelRepository(session)
        entry = await repo.upsert(
            model_id=model_id,
            engine=engine,
            description=description,
            examples=examples,
            replicas=replicas,
            enabled=enabled,
        )
        print(f"[OK] Upserted: {entry!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-yaml", type=Path, help="Path to a models.yaml file to bulk-load")
    mode.add_argument("--add", action="store_true", help="Add/update a single model from CLI args")

    parser.add_argument("--model-id", type=str)
    parser.add_argument("--engine", type=str)
    parser.add_argument("--description", type=str, default="")
    parser.add_argument("--example", action="append", dest="examples", default=[])
    parser.add_argument("--replicas", type=int, default=1)
    parser.add_argument("--disabled", action="store_true", help="Create the model as disabled")

    args = parser.parse_args()

    if args.from_yaml:
        asyncio.run(seed_from_yaml(args.from_yaml))
    else:
        if not args.model_id or not args.engine:
            parser.error("--add requires --model-id and --engine")
        asyncio.run(add_single(
            model_id=args.model_id,
            engine=args.engine,
            description=args.description,
            examples=args.examples,
            replicas=args.replicas,
            enabled=not args.disabled,
        ))


if __name__ == "__main__":
    main()