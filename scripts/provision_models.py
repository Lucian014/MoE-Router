import asyncio
import json
import os
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from database import get_session, ModelRepository

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434",
)

HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "30"))


async def load_catalog() -> list[dict]:
    """
    Load the catalog from the `models` table.
    """
    async with get_session() as session:
        repo = ModelRepository(session)
        entries = await repo.list_enabled()

    if not entries:
        raise ValueError("No enabled models found in the `models` table.")

    return [entry.to_dict() for entry in entries]


def get_installed_ollama_models() -> set[str]:
    """
    Return the names of models currently available in Ollama.
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


async def main() -> None:
    print(f"[*] Ollama: {OLLAMA_URL}")

    models = await load_catalog()
    provision_ollama(models)

    print("[OK] Ollama model provisioning complete.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        sys.exit(1)