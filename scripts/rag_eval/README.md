# rag_eval

Does the information a question needs reach the model? Measured per RAG variant and model size.

| file | role |
|---|---|
| `data/sessions/*.yaml` | the dataset: conversations + questions about old turns, gold = (turn, quote) |
| `dataset.py` | loads and validates it |
| `generate.py` | grows the dataset: you write the user turns (`scenarios/`), a large model the answers |
| `harness.py` | offline: the same `shared/memory` code, cosine in Python instead of pgvector |
| `run.py` | runs the harness (variants x top-k x model profiles, `--ablations`) |
| `judge_retrieval.py` | an LLM judge (GLM, `.env`) grades every retrieved chunk 0/1/2: noise in the top-k, gold check |
| `answer_eval.py` | end to end on the answers: the models answer with the assembled context (production engines), an LLM judge scores them; `answer` where the models are, `judge` where the judge is |
| `cluster_eval.py` | end-to-end through the gateway: seeds the sessions in Postgres, one request per question x variant |
| `report.py` | `results/<run>/report.md` from `results.jsonl` |

```bash
kubectl -n moe-router port-forward svc/moe-ollama 11434:11434 &   # the embedding model
python scripts/rag_eval/run.py
```

Embeddings are cached in `.cache/`; results in `results/` (both git-ignored).
