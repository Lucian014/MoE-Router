# tests

`pytest` from the repository root (`pythonpath = src`). No cluster needed: the classifiers, the
engines and the embedding model are faked; token counts are words where a test says so.

Two groups need live services and skip without them:

| marker | needs | example |
|---|---|---|
| `postgres` | Postgres with pgvector, `POSTGRES_TEST_PORT` | `test_database_memory.py` |
| `rabbitmq` | RabbitMQ, `RABBITMQ_TEST_PORT` | `test_rabbitmq.py` |

```bash
podman run -d --rm --name memory-test-pg -p 55432:5432 -e POSTGRES_USER=test \
    -e POSTGRES_PASSWORD=test -e POSTGRES_DB=model_registry docker.io/pgvector/pgvector:pg16
POSTGRES_TEST_PORT=55432 pytest
```

One test file per module, named after it (`test_assembly.py` -> `shared/memory/assembly.py`).
