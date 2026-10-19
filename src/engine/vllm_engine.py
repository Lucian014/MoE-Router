import httpx

from .base import InferenceEngine


class VLLMEngine(InferenceEngine):
    """
    Talks to vLLM's built-in OpenAI-compatible server
    (`vllm serve <model> --port 8000`), so no vLLM-specific SDK is
    required — just the standard /v1/completions shape.

    Note: unlike Ollama, a vLLM server typically serves exactly one model
    per process. In practice `model` here will usually match whatever
    --served-model-name the server was started with, and `host` will
    point at that model's dedicated Service rather than a shared one.
    """

    def __init__(self, host: str):
        self.host = host

    async def generate(self, model: str, prompt: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{self.host}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": 1024,
                },
            )
            response.raise_for_status()
            data = response.json()
            return data["choices"][0]["text"]

    async def health(self, model: str) -> None:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{self.host}/v1/models")
            response.raise_for_status()
        served = {m.get("id") for m in response.json().get("data", [])}
        if model not in served:
            raise RuntimeError(f"model '{model}' is not served by {self.host} (serving: {sorted(served)})")
