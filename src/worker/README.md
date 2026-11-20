# worker

Generic: one Deployment serves every enabled model of the registry (Postgres `models`).

**Which models.** Every `WORKER_SYNC_SECONDS` (15) the pod re-reads the enabled models and checks
each one's engine (`engine.health`). It consumes `worker.<model>` for every model that answers, on a
channel of its own (`WORKER_PREFETCH` requests per model at once), and stops consuming a model that
stops answering or is disabled. Its queue stays declared, so requests routed to it wait (until
their deadline) instead of being dropped. Where a model runs: `models.inference_host`, or
`OLLAMA_URL` when empty. Adding a model = a registry row (+ `ollama pull`); no manifest.

**One request** (from `worker.<model>`: the queue decides the model):

1. takes the assembled `messages` (or a bare `prompt`) and `num_ctx` / `max_tokens` (else the
   model's own from the registry)
2. `engine.chat(...)` through `src/engine/` (ollama | vllm)
3. stores the turn (W1): session row, turn, routing state, in one transaction; the ORIGINAL prompt,
   not the context
4. publishes `{turn_id}` to `MEMORY_QUEUE` for indexing (fire-and-forget; `--reindex` catches misses)
5. replies to the gateway (`reply_to`, `correlation_id`) with the answer + the routing metadata

A failed store does not fail the answer. The pod is ready once connected to RabbitMQ and the
models were read; an unavailable model only isn't consumed.

File: `worker.py`. Tests: `tests/test_worker.py`.
