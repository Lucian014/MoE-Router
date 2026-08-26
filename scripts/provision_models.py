
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yaml

ROOT = Path(__file__).resolve().parent.parent
CATALOG_PATH = ROOT / "src" / "catalog" / "models.yaml"

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
)

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))


def load_catalog() -> list[dict]:
    if not CATALOG_PATH.exists():
        raise FileNotFoundError(
            f"Catalog not found: {CATALOG_PATH}"
        )

    with open(CATALOG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    models = data.get("models", [])

    if not models:
        raise ValueError(
            "catalog/models.yaml contains no models."
        )

    return models

def get_installed_ollama_models() -> set[str]:
    """
    Return the names of models currently available in Ollama.
    Uses GET /api/tags.
    """
    url = f"{OLLAMA_URL.rstrip('/')}/api/tags"

    request = Request(
        url,
        method="GET",
        headers={
            "Accept": "application/json",
        },
    )

    try:
        with urlopen(request, timeout=HTTP_TIMEOUT) as response:
            data = json.load(response)

    except HTTPError as exc:
        raise RuntimeError(
            f"Ollama /api/tags returned HTTP {exc.code}"
        ) from exc

    except URLError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_URL}: {exc.reason}"
        ) from exc

    models = data.get("models", [])

    return {
        model["name"]
        for model in models
        if model.get("name")
    }

def pull_ollama_model(model_id: str) -> None:
    """
    Pull a model using Ollama POST /api/pull.

    Ollama streams newline-delimited JSON status updates until
    the pull completes.
    """
    url = f"{OLLAMA_URL.rstrip('/')}/api/pull"

    payload = json.dumps({
        "model": model_id,
        "stream": True,
    }).encode("utf-8")

    request = Request(
        url,
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "Accept": "application/x-ndjson",
        },
    )

    print(f"[*] Pulling Ollama model '{model_id}'...")

    try:
        with urlopen(request, timeout=None) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()

                if not line:
                    continue

                status = json.loads(line)

                if "error" in status:
                    raise RuntimeError(
                        f"Ollama failed to pull '{model_id}': "
                        f"{status['error']}"
                    )

                message = status.get("status", "")

                if message:
                    print(f"    {message}")

    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama /api/pull returned HTTP {exc.code}: {body}"
        ) from exc

    except URLError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_URL}: {exc.reason}"
        ) from exc

    print(f"[OK] Model '{model_id}' is ready.")


def provision_ollama(models: list[dict]) -> None:
    installed = get_installed_ollama_models()

    ollama_models = [
        model
        for model in models
        if model.get("engine") == "ollama"
    ]

    print(
        f"[*] Catalog contains {len(ollama_models)} Ollama model(s)."
    )

    for model in ollama_models:
        model_id = model["model_id"]

        if model_id in installed:
            print(
                f"[SKIP] Ollama model already installed: "
                f"{model_id}"
            )
            continue

        pull_ollama_model(model_id)


def main() -> None:
    print(f"[*] Catalog: {CATALOG_PATH}")
    print(f"[*] Ollama: {OLLAMA_URL}")

    models = load_catalog()
    provision_ollama(models)

    print("[OK] Ollama model provisioning complete.")

def delete_ollama_model(model_id: str) -> None:
    """
    Delete an Ollama model using POST/DELETE API endpoint.
    """
    url = f"{OLLAMA_URL.rstrip('/')}/api/delete"

    payload = json.dumps({
        "model": model_id,
    }).encode("utf-8")

    request = Request(
        url,
        data=payload,
        method="DELETE",
        headers={
            "Content-Type": "application/json",
        },
    )

    print(f"[*] Deleting Ollama model '{model_id}'...")

    try:
        with urlopen(request, timeout=HTTP_TIMEOUT) as response:
            if response.status == 200:
                print(f"[OK] Deleted '{model_id}'")
            else:
                raise RuntimeError(
                    f"Unexpected HTTP status: {response.status}"
                )

    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Ollama delete failed for '{model_id}': "
            f"HTTP {exc.code}: {body}"
        ) from exc

    except URLError as exc:
        raise RuntimeError(
            f"Cannot connect to Ollama at {OLLAMA_URL}: {exc.reason}"
        ) from exc


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)