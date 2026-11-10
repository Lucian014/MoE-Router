# memory

The indexing service: turns stored by the workers become searchable chunks.

```
memory.ingest {turn_id} -> index_turn():
    turn from Postgres -> header (extract_question) -> chunks (chunk_turn)
    -> embeddings (bge-m3 on Ollama) + BM25 terms (analyze)
    -> one transaction: old chunks deleted, header set, chunks inserted   (W2)
```

| file | role |
|---|---|
| `service.py` | the consumer: retries (`MEMORY_MAX_ATTEMPTS`, backoff), then `<queue>.dlq`; `--reindex` |
| `indexer.py` | `index_turn`: one turn, idempotent |

Idempotent: a turn whose chunks have the current `CHUNKER_VERSION` and `EMBEDDING_MODEL` is
skipped. Changing either one -> run `--reindex` to rebuild the old turns:

```bash
kubectl -n moe-router exec deploy/moe-memory -- python -m memory.service --reindex [--force]
```

The chunking / embedding / BM25 logic itself is in `src/shared/memory/` (the router and the
evaluation harness use the same code). Tests: `test_memory_indexing.py`, `test_memory_service.py`.
