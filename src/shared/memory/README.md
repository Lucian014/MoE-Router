# shared/memory

The RAG library: pure functions (no I/O except the embedding client), used by the memory service
(writing), the router (reading) and `scripts/rag_eval` (offline). Design: `docs/rag-planv2.md`.

**Writing a turn** (memory service)

| file | does |
|---|---|
| `header.py` | `extract_question`: the turn's header (its question, without pasted code; from the answer if the prompt has no prose) |
| `segment.py` | text -> typed segments (prose, code, log, math, list, table), Markdown parser + line shapes |
| `splitters.py` | a segment > ~400 tokens -> pieces (prose at headings/sentences, lists at items, code at indentation) |
| `chunking.py` | `chunk_turn`: segment -> split -> merge small ones; `CHUNKER_VERSION`, `MIN_TOKENS`, `MAX_TOKENS` |
| `analyzer.py` | `analyze`: BM25 terms (lowercase, no diacritics, identifiers split, Romanian stemmer) |
| `embedding.py` | `EmbeddingClient` interface (`embed`, `health`) + Ollama `/api/embed` |

**Reading for a prompt** (router)

| file | does |
|---|---|
| `header.py` | `build_dense_query`: the prompt's question, for the query vector |
| `presets.py` | V0-V5: which methods run (dense, BM25, contextual, rerank); the router runs V0-V2 |
| `retrieval.py` | `rank`: dense order + BM25 -> RRF -> dedup (newest version wins) -> pool |
| `assembly.py` | `build_messages`: last turn, recent turns, fragments, older turns, within the model's budget |
| `tokens.py` | `count_tokens`: one reference tokenizer (Qwen2.5) for every budget |

Changing anything that shapes chunks (`segment`, `splitters`, `chunking` constants, the header)
means bumping `CHUNKER_VERSION` and reindexing.

Tests: `test_memory_indexing.py`, `test_retrieval.py`, `test_assembly.py`.
