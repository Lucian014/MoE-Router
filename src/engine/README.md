# engine

The worker's only view of an inference server: `InferenceEngine` (`base.py`) with
`chat(model, messages, max_tokens, num_ctx) -> ChatResult` and `health(model)`.

| file | server |
|---|---|
| `ollama_engine.py` | Ollama `/api/chat`; sets `num_ctx` on every request |
| `vllm_engine.py` | vLLM's OpenAI-compatible `/v1/chat/completions`; context fixed by `--max-model-len` |
| `__init__.py` | `get_engine(ENGINE_TYPE, host)` |

A new backend = one adapter here + its name in `_ENGINES` and in
`charts/moe-router/templates/catalog-validate.yaml`. Tests: `tests/test_engines.py`.
