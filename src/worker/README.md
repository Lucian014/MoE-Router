# worker

One Deployment per enabled catalog model (`MODEL_NAME`), consuming `worker.<model>`.

1. takes the assembled `messages` (or a bare `prompt`) and the model's `num_ctx` / `max_tokens`
2. `engine.chat(...)` through `src/engine/` (`ENGINE_TYPE`: ollama | vllm)
3. stores the turn (W1): session row, turn, routing state, in one transaction; the ORIGINAL prompt,
   not the context
4. publishes `{turn_id}` to `MEMORY_QUEUE` for indexing (fire-and-forget; `--reindex` catches misses)
5. replies to the gateway with the answer + the routing metadata

A failed store does not fail the answer. Ready while the engine serves the model (checked every 15 s).

File: `worker.py`. Tests: `tests/test_worker.py`.
