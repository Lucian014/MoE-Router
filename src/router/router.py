"""
router/router.py -- RabbitMQ consumer that picks a model for every prompt.

    Gateway --router.requests--> Router --worker.<model>--> Worker --reply_to--> Gateway

For each request:
  1. two independent Laya classifiers (classifier.py):  category (8)  +  complexity (low / high)
  2. deterministic rule over the catalog (routing_rule.py):
       enabled models of that category -> the one on the predicted tier -> else the other tier
       -> no model in the category: the is_default model
  3. forward to worker.<model> keeping correlation_id + reply_to (the worker answers the Gateway)

The catalog is read from Postgres at start-up and re-read every CATALOG_REFRESH_SECONDS, so
enabling / disabling / adding a model in the DB takes effect without restarting the router.
"""
import asyncio
import json
import sys
import time

from database import ModelRepository, get_session
from shared.config import get_classifier_settings, get_rabbitmq_settings, get_router_settings
from shared.rabbitmq import RabbitMQClient
from shared.utils import model_to_queue

try:
    from router.classifier import CATEGORIES, load_classifiers, predict_complexity, predict_domain
    from router.routing_rule import catalog_from_rows, default_model, select_model, validate_catalog
except ImportError:
    from classifier import CATEGORIES, load_classifiers, predict_complexity, predict_domain
    from routing_rule import catalog_from_rows, default_model, select_model, validate_catalog

# built in main() from shared.config, so importing this module needs no env
mq_client: RabbitMQClient | None = None

_CATALOG: list = []


# ------------------------------------------------------------------ catalog
def _signature(catalog):
    return sorted((m.model_id, m.enabled, m.complexity_tier, m.category, m.is_default)
                  for m in catalog)


def _print_catalog(catalog, warnings):
    print(f"[OK] catalog: {sum(m.enabled for m in catalog)} enabled model(s), "
          f"default = {default_model(catalog).model_id}")
    for c in CATEGORIES:
        low, _ = select_model(catalog, c, "low")
        high, _ = select_model(catalog, c, "high")
        print(f"     {c:17s} low -> {low:32s} high -> {high}")
    for w in warnings:
        print(f"     [warn] {w}")


async def load_router_catalog() -> bool:
    """Read the `models` table, validate it and swap it in. Returns True if it changed.
    Raises on an invalid catalog (no / several default models, two models on the same
    category + tier): at start-up the router refuses to run, during a refresh the previous
    catalog is kept."""
    global _CATALOG
    async with get_session() as session:
        entries = await ModelRepository(session).list_all()
    catalog = catalog_from_rows([e.to_dict() for e in entries])

    errors, warnings = validate_catalog(catalog, CATEGORIES)
    if errors:
        raise ValueError("invalid model catalog:\n  " + "\n  ".join(errors))
    changed = _signature(catalog) != _signature(_CATALOG)
    _CATALOG = catalog
    if changed:
        _print_catalog(catalog, warnings)
    return changed


async def refresh_catalog_forever() -> None:
    while True:
        await asyncio.sleep(get_router_settings().catalog_refresh_seconds)
        try:
            if await load_router_catalog():
                print("[*] catalog changed in the DB -> reloaded")
        except Exception as e:
            print(f"[!] catalog refresh failed, keeping the previous one: {e}")


# ------------------------------------------------------------------ routing
def classify(prompt: str):
    return predict_domain(prompt), predict_complexity(prompt)


async def route_message(message) -> None:
    try:
        payload = json.loads(message.body.decode())
        prompt = payload.get("prompt", "")
        if not prompt:
            raise ValueError("Payload is missing the required 'prompt' field.")

        t0 = time.perf_counter()
        domain, complexity = await asyncio.to_thread(classify, prompt)
        classify_ms = (time.perf_counter() - t0) * 1000

        selected_model, reason = select_model(_CATALOG, domain.category, complexity.level)
        worker_queue = model_to_queue(selected_model)

        payload.update({
            "model": selected_model,
            "routing_reason": reason,                      # match | other-level | fallback:no-model
            "task_type": domain.category,
            "task_type_confidence": round(domain.confidence, 4),
            "task_type_probabilities": {k: round(v, 4) for k, v in domain.probabilities.items()},
            "complexity": complexity.level,
            "complexity_score": round(complexity.score, 4),
            "classify_ms": round(classify_ms, 1),
        })

        top = sorted(domain.probabilities.items(), key=lambda kv: kv[1], reverse=True)[:3]
        print(f"[*] {message.correlation_id}: domain={domain.category} ({domain.confidence:.3f}; "
              + ", ".join(f"{c} {p:.3f}" for c, p in top[1:])
              + f") complexity={complexity.level} (P(high)={complexity.score:.3f}) "
              f"-> {selected_model} [{reason}] -> {worker_queue}  ({classify_ms:.0f} ms)")

        # forward() preserves correlation_id + reply_to
        await mq_client.forward(message=message, routing_key=worker_queue, payload=payload)

    except json.JSONDecodeError:
        print("[!] Received invalid JSON")
        raise
    except Exception as e:
        print(f"[!] Error routing message: {e}")
        raise


async def main():
    global mq_client
    sys.stdout.reconfigure(line_buffering=True)
    rabbit = get_rabbitmq_settings()
    refresh_seconds = get_router_settings().catalog_refresh_seconds
    mq_client = RabbitMQClient.from_settings(rabbit)

    print(f"[*] classifiers on {get_classifier_settings().device}")
    load_classifiers()
    await load_router_catalog()

    await mq_client.connect(service_name="Router")
    await mq_client.channel.set_qos(prefetch_count=1)
    queue = await mq_client.channel.declare_queue(rabbit.middleware_queue, durable=True)

    async def wrapper(message):
        async with message.process():
            await route_message(message)

    await queue.consume(wrapper)
    refresher = asyncio.create_task(refresh_catalog_forever())
    print(f"[*] Router listening on {rabbit.middleware_queue} (catalog refresh every {refresh_seconds}s)")

    try:
        await asyncio.Future()
    finally:
        refresher.cancel()
        await mq_client.close()


if __name__ == "__main__":
    asyncio.run(main())