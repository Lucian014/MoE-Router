#!/usr/bin/env python3
"""
build_manifests.py

Reads catalog/models.yaml (the single source of truth) and generates:

  1. router/models.json          {name, description} projection consumed
                                  by the semantic router's embedding cache.
  2. compute/worker_<name>.yaml  One Kubernetes Deployment per model,
                                  rendered from compute/worker_template.yaml.

Run this any time catalog/models.yaml changes:

    python scripts/build_manifests.py

Nothing else in the system should ever hand-edit the generated files —
they will be silently overwritten on the next run.
"""
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
K8S_DIR = ROOT / "k8s"

sys.path.insert(0, str(SRC_DIR))  # allow `from shared.utils import ...`

from shared.utils import sanitize_k8s_name  # noqa: E402

CATALOG_PATH = SRC_DIR / "catalog" / "models.yaml"
ROUTER_OUTPUT_PATH = SRC_DIR / "router" / "models.json"
TEMPLATE_PATH = ROOT / "scripts" / "templates" / "worker_template.yaml"
COMPUTE_DIR = K8S_DIR / "compute"

REQUIRED_FIELDS = {"model_id", "engine", "description", "replicas"}

# In-cluster host injected into each worker's INFERENCE_HOST env var,
# keyed by engine. This is the ONE place that knows how to reach each
# backend by default — add an entry here when a new engine type is
# onboarded. Per-model overrides (e.g. a dedicated vLLM Service for one
# specific model) can be added as an optional `inference_host` field on
# the catalog entry later without changing this structure.
ENGINE_HOST_DEFAULTS = {
    "ollama": "http://ollama-service.processing.svc.cluster.local:11434",
    "vllm": "http://vllm-service.processing.svc.cluster.local:8000",
}


def load_catalog() -> list[dict]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(f"Catalog not found at {CATALOG_PATH}")

    with open(CATALOG_PATH, "r") as f:
        data = yaml.safe_load(f) or {}

    models = data.get("models", [])
    if not models:
        raise ValueError("catalog/models.yaml contains no models under the 'models' key.")

    seen_names = set()
    seen_k8s_names = {}
    for entry in models:
        missing = REQUIRED_FIELDS - entry.keys()
        if missing:
            raise ValueError(f"Model entry {entry} is missing required fields: {missing}")

        name = entry["model_id"]
        engine = entry["engine"]

        if engine not in ENGINE_HOST_DEFAULTS:
            raise ValueError(
                f"Model '{name}' declares unknown engine '{engine}'. "
                f"Supported engines: {sorted(ENGINE_HOST_DEFAULTS)}. "
                f"(Add it to ENGINE_HOST_DEFAULTS in build_manifests.py and "
                f"a matching adapter in engines/ before using it.)"
            )

        if name in seen_names:
            raise ValueError(f"Duplicate model_id in catalog: {name}")
        seen_names.add(name)

        # Two different model_ids could theoretically sanitize to the same
        # k8s name (e.g. 'foo:1.0' and 'foo:1-0') — catch that early rather
        # than silently overwriting a generated manifest.
        k8s_name = sanitize_k8s_name(name)
        if k8s_name in seen_k8s_names:
            raise ValueError(
                f"'{name}' and '{seen_k8s_names[k8s_name]}' both sanitize to "
                f"the same Kubernetes name '{k8s_name}'. Rename one of them."
            )
        seen_k8s_names[k8s_name] = name

    return models


def build_router_projection(models: list[dict]) -> None:
    """Generate router/models.json with model descriptions and examples."""
    projection = [
        {
            "name": m["model_id"],
            "description": " ".join(m["description"].split()),
            "examples": m.get("examples", []),
        }
        for m in models
    ]

    ROUTER_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(ROUTER_OUTPUT_PATH, "w") as f:
        json.dump(projection, f, indent=2, ensure_ascii=False)

    print(
        f"[OK] Wrote {ROUTER_OUTPUT_PATH.relative_to(ROOT)} "
        f"({len(projection)} models)"
    )


def build_worker_manifests(models: list[dict]) -> None:
    """Generate one compute/worker_<k8s_name>.yaml per catalog entry."""
    if not TEMPLATE_PATH.exists():
        raise FileNotFoundError(f"Template not found at {TEMPLATE_PATH}")

    template = TEMPLATE_PATH.read_text()

    for entry in models:
        model_id = entry["model_id"]
        engine = entry["engine"]
        k8s_name = sanitize_k8s_name(model_id)
        replicas = entry["replicas"]
        inference_host = entry.get("inference_host", ENGINE_HOST_DEFAULTS[engine])

        rendered = (
            template
            .replace("worker-model-name-placeholder", f"worker-{k8s_name}")
            .replace("model-name-placeholder", k8s_name)
            .replace("MODEL_NAME_PLACEHOLDER", model_id)
            .replace("ENGINE_TYPE_PLACEHOLDER", engine)
            .replace("INFERENCE_HOST_PLACEHOLDER", inference_host)
            .replace("REPLICAS_PLACEHOLDER", str(replicas))
        )

        out_path = COMPUTE_DIR / f"worker_{k8s_name}.yaml"
        out_path.write_text(rendered)
        print(f"[OK] Wrote {out_path.relative_to(ROOT)} (replicas={replicas})")


def main():
    models = load_catalog()
    print(f"[*] Loaded {len(models)} model(s) from catalog/models.yaml")
    build_router_projection(models)
    build_worker_manifests(models)
    print("[*] Build complete. Generated files are derived artifacts — do not hand-edit them.")


if __name__ == "__main__":
    main()