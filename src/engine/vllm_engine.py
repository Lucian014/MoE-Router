import httpx

from .base import ChatResult, InferenceEngine, Message


class VLLMEngine(InferenceEngine):
    """
    Talks to vLLM's built-in OpenAI-compatible server
    (`vllm serve <model> --port 8000`), so no vLLM-specific SDK is
    required — just the standard /v1/chat/completions shape. The server
    applies the model's chat template to the messages.

    Note: unlike Ollama, a vLLM server typically serves exactly one model
    per process. In practice `model` here will usually match whatever
    --served-model-name the server was started with, and `host` will
    point at that model's dedicated Service rather than a shared one.
    The context window is fixed by --max-model-len, so `num_ctx` is ignored.
    """

    def __init__(self, host: str, transport: httpx.AsyncBaseTransport | None = None):
        self.host = host
        self._transport = transport     # tests inject an httpx.MockTransport

    def _client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    async def chat(self, model: str, messages: list[Message], max_tokens: int, num_ctx: int) -> ChatResult:
        async with self._client(120.0) as client:
            response = await client.post(
                f"{self.host}/v1/chat/completions",
                json={"model": model, "messages": messages, "max_tokens": max_tokens},
            )
            response.raise_for_status()
            data = response.json()
        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return ChatResult(
            text=(choice.get("message") or {}).get("content") or "",
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            truncated=choice.get("finish_reason") == "length",
        )

    async def health(self, model: str) -> None:
        async with self._client(5.0) as client:
            response = await client.get(f"{self.host}/v1/models")
            response.raise_for_status()
        served = {m.get("id") for m in response.json().get("data", [])}
        if model not in served:
            raise RuntimeError(f"model '{model}' is not served by {self.host} (serving: {sorted(served)})")
