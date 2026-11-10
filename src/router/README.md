# router

Consumes `router.requests`, decides the model and builds its input, forwards to `worker.<model>`.
The worker answers the gateway directly (correlation_id + reply_to are kept).

Per request (`router.py: route_message`):
1. session: routing state + last turns from Postgres (R1)
2. in parallel: domain + complexity, and retrieval (RAG preset V1 / V2)
3. model from the catalog (domain + complexity tier)
4. context in that model's token budget (`shared/memory/assembly.py`)
5. forward, with the routing metadata in the payload

| file | role |
|---|---|
| `router.py` | the consumer: catalog loading/refresh, retrieval, `route_message` |
| `session_routing.py` | pure: classify, or inherit the session's routing (short / ambiguous follow-up) |
| `classifier.py` | the two Laya classifiers (domain, complexity) |
| `classifier_input.py` | pure: what a classifier reads of a long prompt (`edges`: the request + the document's start) |
| `routing_rule.py` | pure: catalog -> model (`match` / `other-level` / `fallback:no-model`), catalog validation |
| `domain_schema.py`, `complexity_schema.py` | the classifiers' questions and labels (must match training) |
| `download_checkpoints.py` | init container: fetches the private checkpoints from Hugging Face |
| `checkpoints/` | local copies of the checkpoints (git-ignored) |

The catalog is the `models` table, re-read every `CATALOG_REFRESH_SECONDS`. A retrieval failure
falls back to V0 (recent turns only); the request still gets an answer.

Tests: `test_router_session.py`, `test_session_routing.py`, `test_routing_rule.py`, `test_classifier_input.py`.
