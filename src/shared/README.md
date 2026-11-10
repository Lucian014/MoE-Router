# shared

Code every service image copies. Depends on no service.

| file | role |
|---|---|
| `config.py` | every runtime variable (pydantic-settings), one lazy getter per group; the env table is at the top; the defaults other code needs too (`DEFAULT_CONTEXT_WINDOW`, `DEFAULT_EMBEDDING_MODEL`, `EMBEDDING_DIM`, checkpoint names, ...) |
| `rabbitmq.py` | `RabbitMQClient`: RPC call / reply / forward with a deadline header, dead-letter queues, drain on shutdown |
| `lifecycle.py` | start-up `retry`, SIGTERM -> stop event, readiness file |
| `prompts.py` | `SYSTEM_PROMPT` |
| `utils.py` | `model_to_queue`: model_id -> `worker.<k8s-safe name>` (the chart uses the same rule); `model_is_listed`: Ollama's `:latest` rule |
| `memory/` | the conversation-memory library, see `memory/README.md` |
