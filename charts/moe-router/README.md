# charts/moe-router

The whole platform as one Helm chart (deployed by `skaffold run`, namespace `moe-router`).

| template | what |
|---|---|
| `gateway/`, `router/`, `memory/` | the services (router: init container that downloads the classifiers) |
| `workers/` | one Deployment per enabled model of `catalog.models`, queue `worker.<name>` |
| `ollama/`, `postgres/` (pgvector), `rabbitmq/` | the infrastructure; can be disabled for external ones |
| `setup-job.yaml` | per install / upgrade: migrations -> catalog seed -> model pulls -> re-seed |
| `catalog-configmap.yaml` | `catalog.models` as the `models.yaml` the setup Job seeds |
| `catalog-validate.yaml` | fails the render on an invalid catalog (engine, default model, ...) |
| `configmap.yaml` | the shared env (queues, hosts, RAG / classifier settings) |
| `_helpers.tpl` | names, images, `k8sName` (model_id -> resource / queue name) |

`values.yaml` is the place to change anything: the catalog, resources, RAG settings.
`values/values-virtualbox.yaml` overrides it for the VM setup; `ci/` holds the CI values and smoke test.
