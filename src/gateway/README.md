# gateway

FastAPI, the only service clients talk to. Stateless: no database, no model.

Every client endpoint is under `/api`, so the UI's own pages can use any other path.

- `POST /api/generate` `{prompt, session_id?, variant?}` -> publishes to `MIDDLEWARE_QUEUE` (RPC) and
  waits up to `RPC_TIMEOUT_SECONDS` for the worker's reply. Without a `session_id` a new one is
  made; the reply always carries it, so the client sends it back on the next turn.
- `variant` (V0-V9) picks the RAG preset, only when the router has `ALLOW_VARIANT_OVERRIDE` (evaluation).
- `POST /generate`: the same endpoint at its pre-`/api` path (deprecated), for existing scripts and curl.
- `/health` (process up), `/ready` (connected to RabbitMQ): at the root, for the Kubernetes probes.
- Errors: 504 timeout, 502 the router / worker answered with an error.
- `/` and everything else: the built interface (`src/ui`). The Dockerfile builds it in a Node stage
  and copies `dist/` to `./static`; `mount_ui` mounts it after every route, so `/api/*` and the
  probes always win. Files under `assets/` (hashed names) are cached forever, `index.html` is
  revalidated. Without `./static` (tests, local runs) there is simply no UI.

File: `main.py`. Tests: `tests/test_gateway.py`.
