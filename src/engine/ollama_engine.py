import httpx

from .base import InferenceEngine


class OllamaEngine(InferenceEngine):
    """Talks to an Ollama daemon's native /api/generate endpoint."""

    def __init__(self, host: str):
        self.host = host

    async def generate(self, model: str, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.host}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
            )
            response.raise_for_status()
            return response.json().get("response", "")
