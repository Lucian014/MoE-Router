import httpx

from .base import ChatResult, InferenceEngine, Message


def model_is_listed(tags: dict, model: str) -> bool:
    """True if `model` is in an Ollama /api/tags response. A model pulled without a tag
    is listed as '<name>:latest'."""
    names = {m.get("name") for m in tags.get("models", [])}
    return model in names or (":" not in model and f"{model}:latest" in names)


class OllamaEngine(InferenceEngine):
    """Talks to an Ollama daemon's native /api/chat endpoint."""

    def __init__(self, host: str, transport: httpx.AsyncBaseTransport | None = None):
        self.host = host
        self._transport = transport     # tests inject an httpx.MockTransport

    def _client(self, timeout: float) -> httpx.AsyncClient:
        return httpx.AsyncClient(timeout=timeout, transport=self._transport)

    async def chat(self, model: str, messages: list[Message], max_tokens: int, num_ctx: int) -> ChatResult:
        # num_ctx is set on every request: Ollama's own default is small, and a prompt longer
        # than the context is cut from the start without an error
        async with self._client(240.0) as client:
            response = await client.post(
                f"{self.host}/api/chat",
                json={
                    "model": model,
                    "messages": messages,
                    "stream": False,
                    "options": {"num_ctx": num_ctx, "num_predict": max_tokens},
                },
            )
            response.raise_for_status()
            data = response.json()
        return ChatResult(
            text=(data.get("message") or {}).get("content", ""),
            prompt_tokens=data.get("prompt_eval_count"),
            completion_tokens=data.get("eval_count"),
            truncated=data.get("done_reason") == "length",
        )

    async def health(self, model: str) -> None:
        async with self._client(5.0) as client:
            response = await client.get(f"{self.host}/api/tags")
            response.raise_for_status()
        if not model_is_listed(response.json(), model):
            raise RuntimeError(f"model '{model}' is not pulled on {self.host}")
