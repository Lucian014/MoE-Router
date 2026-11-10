# gateway

FastAPI, the only service clients talk to. Stateless: no database, no model.

- `POST /generate` `{prompt, session_id?, variant?}` -> publishes to `MIDDLEWARE_QUEUE` (RPC) and
  waits up to `RPC_TIMEOUT_SECONDS` for the worker's reply. Without a `session_id` a new one is
  made; the reply always carries it, so the client sends it back on the next turn.
- `variant` (V0-V9) picks the RAG preset, only when the router has `ALLOW_VARIANT_OVERRIDE` (evaluation).
- `/health` (process up), `/ready` (connected to RabbitMQ).
- Errors: 504 timeout, 502 the router / worker answered with an error.

File: `main.py`. Tests: `tests/test_gateway.py`.
