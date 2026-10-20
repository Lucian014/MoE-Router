# Database layer (PostgreSQL + SQLAlchemy + Alembic)

## Layout

```
database/
├── __init__.py         # re-exports shared infra + every table's public API
├── config.py            # env-var driven Postgres connection settings
├── session.py            # async engine + get_session() — shared by ALL tables
├── base.py               # the ONE shared declarative Base every schema.py uses
├── alembic.ini
├── requirements.txt
├── dbctl.sh             # port-forwards to the in-cluster Postgres, pulls
│                            # credentials from postgres-secret, then runs
│                            # whatever alembic command you pass it
├── migrations/
│   ├── env.py             # wires Alembic to database.config + database.models
│   ├── script.py.mako
│   └── versions/
│       └── 0001_create_models_table.py
└── models/                 # one subfolder PER TABLE
    ├── __init__.py           # imports every table's schema.py (see below)
    └── model_registry/        # <- the "models" table
        ├── __init__.py          # exposes this table's schema + repository
        ├── schema.py             # ORM class: ModelRegistryEntry
        └── repository.py         # ModelRepository: CRUD + queries
```

## What's shared vs. what's per-table

**Shared once, for the whole project — never duplicated:**
- `config.py` — connection settings
- `session.py` — the engine and `get_session()`
- `base.py` — the declarative `Base` class
- `alembic.ini` + `migrations/env.py` + `migrations/script.py.mako` — the
  migration mechanism itself

**Added once per new table**, inside `models/<table_name>/`:
- `schema.py` — the ORM class (columns, types, defaults)
- `repository.py` — a `<Thing>Repository` class with the CRUD/query methods
  for that table only
- `__init__.py` — exposes the two above

## Adding a new table

1. Create the folder: `database/models/<table_name>/`
2. Write `schema.py`:
   ```python
   from database.base import Base
   from sqlalchemy.orm import Mapped, mapped_column
   from sqlalchemy import Integer, String

   class YourThing(Base):
       __tablename__ = "your_things"
       id: Mapped[int] = mapped_column(Integer, primary_key=True)
       name: Mapped[str] = mapped_column(String(255), nullable=False)
   ```
3. Write `repository.py` following the pattern in
   `model_registry/repository.py` (takes an `AsyncSession` in `__init__`,
   exposes `get_by_id`, `list_all`, `create`, etc. — only `flush()`, never
   `commit()`, since `get_session()` owns the transaction).
4. Write `__init__.py` exporting the schema class + repository class.
5. Register it so Alembic can see it — add one line to
   `database/models/__init__.py`:
   ```python
   from database.models.your_table.schema import YourThing  # noqa: F401
   ```
6. Generate and review the migration:
   ```bash
   ./dbctl.sh revision --autogenerate -m "create your_things table"
   ```
   **Read the generated file before applying it** — autogenerate can get
   renames wrong (emitting a drop+add instead of a rename), which loses
   data silently if applied without review.
7. Apply it:
   ```bash
   ./dbctl.sh upgrade head
   ```
8. (Optional) Re-export from the root `database/__init__.py` if you want
   `from database import YourThingRepository` to work.

Step 5 is the one easy thing to forget — if a new schema module is never
imported anywhere, it never registers onto `Base.metadata`, and
`alembic revision --autogenerate` will not know it exists.

## Schema: `model_registry` table (`models`)

| column       | type          | notes                              |
|--------------|---------------|-------------------------------------|
| id           | serial, PK    | internal surrogate key              |
| model_id     | varchar(255), unique | natural key, e.g. `"qwen2.5:7b"` |
| engine       | varchar(64)   | e.g. `"ollama"`                     |
| description  | text          | fed to the embedding model          |
| examples     | text[]        | fed to the embedding model          |
| replicas     | integer, default 1 |                                |
| enabled      | boolean, default true | router should only load `enabled` |
| created_at   | timestamptz    | server-generated                    |
| updated_at   | timestamptz    | server-generated, updates on write  |

## Environment variables

Defined in `src/shared/config.py` (PostgresSettings). In Kubernetes the Helm
chart sets them: `POSTGRES_HOST`/`POSTGRES_PORT` from the `<release>-config`
ConfigMap, and `POSTGRES_DB`/`POSTGRES_USER`/`POSTGRES_PASSWORD` from the
generated `<release>-postgres` Secret (charts/moe-router/templates/postgres/).

- `POSTGRES_HOST` (default `postgres`; the chart sets `<release>-postgres`)
- `POSTGRES_PORT` (default `5432`)
- `POSTGRES_DB` (default `model_registry`)
- `POSTGRES_USER` — **no default, must be set**
- `POSTGRES_PASSWORD` — **no default, must be set**

`POSTGRES_USER`/`POSTGRES_PASSWORD` have no fallback on purpose — the app
will fail loudly rather than silently connect with a hardcoded credential.
`dbctl.sh` handles this for you automatically (see below) by reading
both live from `postgres-secret`.

Or set `ASYNC_DATABASE_URL` / `SYNC_DATABASE_URL` directly to override
everything above at once.

**Wiring the Secret into a Deployment** (e.g. once the router is migrated):

```yaml
env:
  - name: POSTGRES_DB
    valueFrom:
      secretKeyRef: { name: postgres-secret, key: POSTGRES_DB }
  - name: POSTGRES_USER
    valueFrom:
      secretKeyRef: { name: postgres-secret, key: POSTGRES_USER }
  - name: POSTGRES_PASSWORD
    valueFrom:
      secretKeyRef: { name: postgres-secret, key: POSTGRES_PASSWORD }
  - name: POSTGRES_HOST
    value: "postgres"  # same-namespace Service DNS name
```

## Running migrations

**From outside the cluster** (e.g. your laptop), use `dbctl.sh` — it
opens a `kubectl port-forward` to the in-cluster Postgres, pulls
`POSTGRES_DB`/`USER`/`PASSWORD` live from `postgres-secret` (the same
Secret the pod itself consumes — nothing is duplicated or hardcoded), runs
whatever alembic command you pass it, then tears the tunnel down when it's
done:

```bash
cd database
pip install -r requirements.txt

./dbctl.sh current                                 # show applied revision
./dbctl.sh history                                  # show full migration chain
./dbctl.sh upgrade head                              # apply all pending migrations
./dbctl.sh downgrade -1                                # roll back one migration
./dbctl.sh revision --autogenerate -m "message"         # generate a new migration
```

Anything after `./dbctl.sh` is forwarded as-is to `alembic`, so any
Alembic subcommand works the same way.

**From inside the cluster** (or with `POSTGRES_*` env vars / a tunnel
already set up manually), you can call `alembic` directly instead:

```bash
cd database
alembic upgrade head
```

## Using a repository

```python
from database import get_session, ModelRepository

async def example():
    async with get_session() as session:
        repo = ModelRepository(session)

        await repo.create(
            model_id="qwen2.5-coder:7b",
            engine="ollama",
            description="Specialized coding model...",
            examples=["Write a Python function that parses a JSON file..."],
            replicas=1,
            enabled=True,
        )

        enabled_models = await repo.list_enabled()
```