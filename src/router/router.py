import asyncio
import json
import os
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from shared.rabbitmq import RabbitMQClient
from shared.utils import model_to_queue

MIDDLEWARE_QUEUE = os.getenv("MIDDLEWARE_QUEUE", "router.requests")
MODELS_JSON_PATH = Path(os.getenv("MODELS_JSON_PATH", "router/models.json"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2")

mq_client = RabbitMQClient(
    host=os.getenv("RABBITMQ_HOST", "localhost"),
    user=os.getenv("RABBITMQ_USER", "guest"),
    password=os.getenv("RABBITMQ_PASS", "guest"),
)

# Populated once at startup by load_router_catalog(), then read-only for
# the life of the process -> safe to share across concurrent handlers.
_EMBEDDER: SentenceTransformer | None = None
_MODEL_NAMES: list[str] = []
_MODEL_DESCRIPTION_EMBEDDINGS: np.ndarray | None = None
_MODEL_EXAMPLE_EMBEDDINGS: np.ndarray | None = None


def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize row vectors so a plain dot product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms


def load_router_catalog() -> None:
    """
    Load router/models.json and cache normalized embeddings for both
    model descriptions and examples.

    Description and examples are embedded separately so routing can
    combine semantic similarity to the model's general capabilities
    with similarity to concrete example prompts.
    """
    global _EMBEDDER
    global _MODEL_NAMES
    global _MODEL_DESCRIPTION_EMBEDDINGS
    global _MODEL_EXAMPLE_EMBEDDINGS

    if not MODELS_JSON_PATH.exists():
        raise FileNotFoundError(
            f"{MODELS_JSON_PATH} not found. "
            "Run `python scripts/build_manifests.py` first."
        )

    with open(MODELS_JSON_PATH, "r", encoding="utf-8") as f:
        catalog = json.load(f)

    if not catalog:
        raise ValueError(
            f"{MODELS_JSON_PATH} is empty — router has no models to choose from."
        )

    print(f"[*] Loading embedding model '{EMBEDDING_MODEL_NAME}' (CPU)...")
    _EMBEDDER = SentenceTransformer(
        EMBEDDING_MODEL_NAME,
        device="cpu",
    )

    _MODEL_NAMES = [entry["name"] for entry in catalog]

    descriptions = [
        entry.get("description", "")
        for entry in catalog
    ]

    # Encode descriptions.
    raw_description_embeddings = _EMBEDDER.encode(
        descriptions,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    _MODEL_DESCRIPTION_EMBEDDINGS = (
        _normalize(raw_description_embeddings)
        .astype(np.float32)
    )

    # Encode examples.
    #
    # We allow different numbers of examples per model. To keep the
    # resulting matrix rectangular, models with fewer examples are
    # padded with zero vectors.
    max_examples = max(
        len(entry.get("examples", []))
        for entry in catalog
    )

    if max_examples == 0:
        # No examples at all -> use an empty example matrix.
        embedding_dim = _MODEL_DESCRIPTION_EMBEDDINGS.shape[1]
        _MODEL_EXAMPLE_EMBEDDINGS = np.empty(
            (len(catalog), 0, embedding_dim),
            dtype=np.float32,
        )
    else:
        embedding_dim = _MODEL_DESCRIPTION_EMBEDDINGS.shape[1]

        example_embeddings = np.zeros(
            (len(catalog), max_examples, embedding_dim),
            dtype=np.float32,
        )

        for model_idx, entry in enumerate(catalog):
            examples = entry.get("examples", [])

            if not examples:
                continue

            raw_examples = _EMBEDDER.encode(
                examples,
                convert_to_numpy=True,
                show_progress_bar=False,
            )

            normalized_examples = _normalize(raw_examples).astype(np.float32)

            example_embeddings[
                model_idx,
                :len(examples),
                :
            ] = normalized_examples

        _MODEL_EXAMPLE_EMBEDDINGS = example_embeddings

    print(
        f"[OK] Cached embeddings for {len(_MODEL_NAMES)} model(s): "
        f"{_MODEL_NAMES}"
    )


def select_model(prompt: str) -> tuple[str, float]:
    """
    Embed the incoming prompt and select the model using both:

      - model description similarity: 40%
      - closest example similarity:   60%

    Returns:
        (selected_model, confidence_score)
    """
    if (
        _EMBEDDER is None
        or _MODEL_DESCRIPTION_EMBEDDINGS is None
        or _MODEL_EXAMPLE_EMBEDDINGS is None
    ):
        raise RuntimeError(
            "Router catalog not loaded. "
            "Call load_router_catalog() first."
        )

    prompt_embedding = _EMBEDDER.encode(
        [prompt],
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    prompt_embedding = _normalize(prompt_embedding).astype(np.float32)[0]

    description_similarities = (
        _MODEL_DESCRIPTION_EMBEDDINGS @ prompt_embedding
    )

    if _MODEL_EXAMPLE_EMBEDDINGS.shape[1] > 0:
        example_similarities = np.einsum(
            "med,d->me",
            _MODEL_EXAMPLE_EMBEDDINGS,
            prompt_embedding,
        )

        closest_example_similarities = np.max(
            example_similarities,
            axis=1,
        )
    else:
        closest_example_similarities = np.zeros(
            len(_MODEL_NAMES),
            dtype=np.float32,
        )

    routing_score = (
        0.4 * description_similarities
        + 0.6 * closest_example_similarities
    )

    best_idx = int(np.argmax(routing_score))

    selected_model = _MODEL_NAMES[best_idx]
    routing_score = float(routing_score[best_idx])

    print(
        f"[*] Semantic match: '{selected_model}' "
        f"(score={routing_score:.4f}, "
        f"description={description_similarities[best_idx]:.4f}, "
        f"example={closest_example_similarities[best_idx]:.4f})"
    )

    return selected_model, routing_score


async def route_message(message) -> None:
    """Receive a request from the Gateway, semantically pick a model, forward it."""
    try:
        payload = json.loads(message.body.decode())
        prompt = payload.get("prompt", "")

        if not prompt:
            raise ValueError("Payload is missing the required 'prompt' field.")

        selected_model, routing_score = select_model(prompt)
        worker_queue = model_to_queue(selected_model)

        payload["model"] = selected_model
        payload["routing_score"] = routing_score

        print(f"[*] Routing request {message.correlation_id} -> model={selected_model} -> {worker_queue}")

        # forward() preserves correlation_id + reply_to, so the Worker
        # can reply directly to the Gateway's private callback queue.
        await mq_client.forward(message=message, routing_key=worker_queue, payload=payload)

    except json.JSONDecodeError:
        print("[!] Received invalid JSON")
        raise
    except Exception as e:
        print(f"[!] Error routing message: {e}")
        raise


async def main():
    sys.stdout.reconfigure(line_buffering=True)

    load_router_catalog()

    await mq_client.connect(service_name="Semantic Router")
    print(f"[*] Router listening on {MIDDLEWARE_QUEUE}")

    await mq_client.channel.set_qos(prefetch_count=1)
    queue = await mq_client.channel.declare_queue(MIDDLEWARE_QUEUE, durable=True)

    async def wrapper(message):
        async with message.process():
            await route_message(message)

    await queue.consume(wrapper)
    print(f"[*] Waiting for requests on {MIDDLEWARE_QUEUE}")

    try:
        await asyncio.Future()
    finally:
        await mq_client.close()


if __name__ == "__main__":
    asyncio.run(main())