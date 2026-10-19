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

Lifecycle (shared/lifecycle.py): start-up waits (with backoff) for a valid catalog and for
RabbitMQ; the readiness file exists while the router is consuming; on SIGTERM it stops
consuming, finishes the in-flight message (SHUTDOWN_GRACE_SECONDS) and closes.
"""
import asyncio
import json
import sys
import time

from database import ModelRepository, dispose_engine, get_session
from shared.config import get_classifier_settings, get_lifecycle_settings, get_rabbitmq_settings, get_router_settings
from shared.lifecycle import Stopped, install_stop_signals, retry, set_ready
from shared.rabbitmq import RabbitMQClient, remaining_seconds
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
    category + tier): at start-up the router retries until the catalog is valid (and stays
    not ready meanwhile), during a refresh the previous catalog is kept."""
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

        # forward() preserves correlation_id + reply_to (+ the deadline)
        if not await mq_client.forward(message=message, routing_key=worker_queue, payload=payload):
            print(f"[!] {message.correlation_id}: deadline passed during routing, dropped")

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
    lifecycle = get_lifecycle_settings()
    refresh_seconds = get_router_settings().catalog_refresh_seconds
    mq_client = RabbitMQClient.from_settings(rabbit)
    stop = install_stop_signals()
    set_ready(lifecycle.ready_file, False)

    print(f"[*] classifiers on {get_classifier_settings().device}")
    load_classifiers()
    try:
        # Postgres may not be up yet, or the migrations / seed may not have run yet:
        # wait for a valid catalog instead of crash-looping.
        await retry(load_router_catalog, "loading the model catalog", stop_event=stop)
        await mq_client.connect(service_name="Router", stop_event=stop)
    except Stopped:
        print("[*] stopped before start-up finished")
        return

    async def wrapper(message):
        async with message.process():
            left = remaining_seconds(message)
            if left is not None and left <= 0:
                print(f"[!] {message.correlation_id}: deadline passed, dropping (caller gave up)")
                return
            try:
                await route_message(message)
            except Exception as e:
                # Answer the Gateway right away instead of letting it wait for its timeout.
                # The message is still acked: a request that can't be routed would fail
                # the same way on every redelivery.
                await mq_client.reply_error(message, f"routing failed: {e}")

    await mq_client.consume_messages(rabbit.middleware_queue, wrapper, prefetch_count=1)
    refresher = asyncio.create_task(refresh_catalog_forever())
    set_ready(lifecycle.ready_file, True)
    print(f"[*] Router listening on {rabbit.middleware_queue} (catalog refresh every {refresh_seconds}s)")

    await stop.wait()
    print("[*] SIGTERM received, shutting down")
    set_ready(lifecycle.ready_file, False)
    refresher.cancel()
    await mq_client.drain(lifecycle.shutdown_grace_seconds)
    await mq_client.close()
    await dispose_engine()
    print("[OK] router stopped")


if __name__ == "__main__":
    asyncio.run(main())