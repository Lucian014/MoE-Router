# MoE-Router

Routes every chat prompt to a small expert model picked by two classifiers (domain + complexity),
and gives that model the conversation's memory (recent turns + retrieved fragments) within its
token budget. Runs on Kubernetes (k3s), one Helm chart.

```
client -> gateway -> RabbitMQ -> router -> worker.<model> -> Ollama / vLLM -> reply to the gateway
                                   |  classify + retrieve        | store the turn
                                   v                             v
                                Postgres  <---- memory service <-- memory.ingest {turn_id}
                           (models, sessions, turns, chunks + pgvector)
```

| folder | what it is |
|---|---|
| `src/gateway/` | HTTP API (`POST /generate`), the only public entry point |
| `src/router/` | classifiers + model choice + context assembly |
| `src/worker/` | one per model: calls the engine, stores the turn |
| `src/memory/` | indexes stored turns into chunks (embeddings + BM25 terms) |
| `src/engine/` | adapters to the inference servers (Ollama, vLLM) |
| `src/shared/` | settings, RabbitMQ client, lifecycle, the memory library (`shared/memory/`) |
| `src/database/` | SQLAlchemy tables, repositories, Alembic migrations |
| `src/catalog/models.yaml` | the model catalog for local seeding (the chart has its own copy in `values.yaml`) |
| `charts/moe-router/` | the Helm chart: every service, Postgres, RabbitMQ, Ollama, the setup Job |
| `scripts/` | seeding, model pulls, classifier training, evaluations |
| `tests/` | pytest; some need a live Postgres / RabbitMQ |
| `docs/rag-planv2.md` | the design of sessions, memory and RAG |
| `legacy/` | the earlier routers, kept for reference, not used |

Every runtime variable is defined once, in `src/shared/config.py` (table at the top of the file).

```bash
skaffold run                                   # build the images + deploy the chart (namespace moe-router)
pytest                                         # tests (Postgres / RabbitMQ ones skip without them)
ruff check src tests scripts
```
