#!/usr/bin/env python3
"""
build_manifests.py

Reads the model catalog from PostgreSQL (via ModelRepository) — the single
source of truth — and generates:

  1. compute/worker_<name>.yaml  One Kubernetes Deployment per model,
                                  rendered from compute/worker_template.yaml.

Run this any time the `models` table changes:

    python scripts/build_manifests.py

The `models` table no longer has a `replicas` column: workers are autoscaled by KEDA on their
`worker.<model>` queue, so every Deployment starts with WORKER_REPLICAS pods (env, default 1).

Nothing else in the system should ever hand-edit the generated files —
they will be silently overwritten on the next run.

"""
import asyncio
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
K8S_DIR = ROOT / "k8s"

sys.path.insert(0, str(SRC_DIR))

from database import get_session, ModelRepository  # noqa: E402
from shared.utils import sanitize_k8s_name  # noqa: E402

TEMPLATE_PATH = ROOT / "scripts" / "templates" / "worker_template.yaml"
COMPUTE_DIR = K8S_DIR / "compute"
WORKER_REPLICAS = int(os.getenv("WORKER_REPLICAS", "1"))

ENGINE_HOST_DEFAULTS = {
    "ollama": "http://ollama-service.processing.svc.cluster.local:11434",
    "vllm": "http://vllm-service.processing.svc.cluster.local:8000",
}


async def load_catalog() -> list[dict]:
    """Load the catalog from the `models` table (ModelRegistryEntry.to_dict() rows)."""
    async with get_session() as session:
        repo = ModelRepository(session)
        entries = await repo.list_all()

    if not entries:
        raise ValueError("The `models` table is empty — nothing to build manifests for.")

    models = [entry.to_dict() for entry in entries]

    seen_k8s_names: dict[str, str] = {}
    for entry in models:
        name = entry["model_id"]
        engine = entry["engine"]

        if engine not in ENGINE_HOST_DEFAULTS:
            raise ValueError(
                f"Model '{name}' declares unknown engine '{engine}'. "
                f"Supported engines: {sorted(ENGINE_HOST_DEFAULTS)}. "
                f"(Add it to ENGINE_HOST_DEFAULTS in build_manifests.py and "
                f"a matching adapter in engines/ before using it.)"
            )

        k8s_name = sanitize_k8s_name(name)
        if k8s_name in seen_k8s_names:
            raise ValueError(
                f"'{name}' and '{seen_k8s_names[k8s_name]}' both sanitize to "
                f"the same Kubernetes name '{k8s_name}'. Rename one of them."
            )
        seen_k8s_names[k8s_name] = name

    return models


def build_worker_manifests(models: list[dict]) -> None:
    """Generate one compute/worker_<k8s_name>.yaml per catalog entry."""
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template not found at {TEMPLATE_PATH}")

    template = TEMPLATE_PATH.read_text()

    for entry in models:
        model_id = entry["model_id"]
        engine = entry["engine"]
        k8s_name = sanitize_k8s_name(model_id)
        inference_host = entry.get("inference_host") or ENGINE_HOST_DEFAULTS[engine]

        rendered = (
            template
            .replace("worker-model-name-placeholder", f"worker-{k8s_name}")
            .replace("model-name-placeholder", k8s_name)
            .replace("MODEL_NAME_PLACEHOLDER", model_id)
            .replace("ENGINE_TYPE_PLACEHOLDER", engine)
            .replace("INFERENCE_HOST_PLACEHOLDER", inference_host)
            .replace("REPLICAS_PLACEHOLDER", str(WORKER_REPLICAS))
        )

        out_path = COMPUTE_DIR / f"worker_{k8s_name}.yaml"
        out_path.write_text(rendered)
        print(f"[OK] Wrote {out_path.relative_to(ROOT)} "
              f"(category={entry['category']}, tier={entry['complexity_tier']}, "
              f"enabled={entry['enabled']})")


async def main():
    models = await load_catalog()
    print(f"[*] Loaded {len(models)} model(s) from the `models` table")
    build_worker_manifests(models)
    print("[*] Build complete. Generated files are derived artifacts — do not hand-edit them.")


if __name__ == "__main__":
    asyncio.run(main())