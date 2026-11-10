"""Embedding clients (docs/rag-planv2.md 4.4). Same idea as engine.InferenceEngine: the memory
service, the router and the harness talk to this interface, so the model or the server (Ollama now,
a GPU server later) can change without touching them. Changing the model means reindexing:
every chunk row records the embedding_model it was produced with."""
from abc import ABC, abstractmethod

import httpx

from shared.config import DEFAULT_EMBEDDING_MODEL
from shared.utils import model_is_listed


class EmbeddingClient(ABC):
    model: str

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """One vector per text, in order."""
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> None:
        """Return if the server is up and has the model; raise (with the reason) otherwise."""
        raise NotImplementedError


class OllamaEmbeddingClient(EmbeddingClient):
    """Ollama /api/embed, one call per batch. truncate=False: a text longer than the model's
    context is an error, not silently cut at the end (callers size their inputs: chunks are
    ~400 tokens, the dense query at most 512)."""

    def __init__(self, host: str, model: str = DEFAULT_EMBEDDING_MODEL, keep_alive: str | None = None,
                 transport: httpx.AsyncBaseTransport | None = None, timeout: float = 120.0):
        self.host, self.model, self.keep_alive = host, model, keep_alive
        self._transport, self._timeout = transport, timeout

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        body = {"model": self.model, "input": texts, "truncate": False}
        if self.keep_alive:
            body["keep_alive"] = self.keep_alive
        async with httpx.AsyncClient(timeout=self._timeout, transport=self._transport) as client:
            response = await client.post(f"{self.host}/api/embed", json=body)
            response.raise_for_status()
            vectors = response.json().get("embeddings") or []
        if len(vectors) != len(texts):
            raise RuntimeError(f"{self.model}: asked for {len(texts)} embeddings, got {len(vectors)}")
        return vectors

    async def health(self) -> None:
        async with httpx.AsyncClient(timeout=5.0, transport=self._transport) as client:
            response = await client.get(f"{self.host}/api/tags")
            response.raise_for_status()
        if not model_is_listed(response.json(), self.model):
            raise RuntimeError(f"embedding model '{self.model}' is not pulled on {self.host}")
