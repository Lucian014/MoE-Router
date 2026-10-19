import httpx

from .base import InferenceEngine


def model_is_listed(tags: dict, model: str) -> bool:
    """True if `model` is in an Ollama /api/tags response. A model pulled without a tag
    is listed as '<name>:latest'."""
    names = {m.get("name") for m in tags.get("models", [])}
    return model in names or (":" not in model and f"{model}:latest" in names)


class OllamaEngine(InferenceEngine):
    """Talks to an Ollama daemon's native /api/generate endpoint."""

    def __init__(self, host: str):
        self.host = host

    async def generate(self, model: str, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=240.0) as client:
            response = await client.post(
                f"{self.host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json().get("response", "")

    async def health(self, model: str) -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{self.host}/api/tags")
            response.raise_for_status()
        if not model_is_listed(response.json(), model):
            raise RuntimeError(f"model '{model}' is not pulled on {self.host}")
