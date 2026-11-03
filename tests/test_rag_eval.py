"""The evaluation harness (scripts/rag_eval/): the dataset is valid, gold quotes map to chunks, and the
harness runs end to end with a fake embedding model (no Ollama)."""
import asyncio
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "rag_eval"))

from dataset import Gold, Turn, check, gold_ords, load_dataset  # noqa: E402
from harness import (  # noqa: E402
    CachedEmbedder,
    ModelProfile,
    ablation_configs,
    evaluate,
    index_sessions,
    main_configs,
)
from report import write_report  # noqa: E402

from shared.memory.analyzer import analyze  # noqa: E402
from shared.memory.chunking import chunk_turn  # noqa: E402

SESSIONS = load_dataset()


class BagOfWordsEmbedder:
    """Deterministic stand-in for bge-m3: hashed bag of terms, so related texts are close."""
    model = "bag-of-words"

    def __init__(self):
        self.calls = 0

    async def embed(self, texts):
        self.calls += len(texts)
        out = []
        for t in texts:
            v = [0.0] * 256
            for term in analyze(t):
                v[int(hashlib.md5(term.encode()).hexdigest(), 16) % 256] += 1.0
            out.append(v)
        return out


@pytest.mark.parametrize("session", SESSIONS, ids=[s.id for s in SESSIONS])
def test_dataset_is_valid(session):
    assert check(session) == []
    assert session.questions


def test_gold_maps_to_the_overlapping_chunks():
    answer = "Intro paragraph.\n\n```python\ndef f():\n    return 1\n```\n\nClosing words."
    turn = Turn("short question?", answer)
    drafts = chunk_turn(turn.user, turn.assistant)
    ords = gold_ords(turn, drafts, Gold(1, "return 1"))
    assert ords and all("return 1" in drafts[o].content for o in ords)
    assert gold_ords(turn, drafts, Gold(1, "not there")) == []


def test_harness_runs_end_to_end(tmp_path):
    sessions = [s for s in SESSIONS if s.id.startswith(("s01", "s03"))]
    fake = BagOfWordsEmbedder()
    embedder = CachedEmbedder(fake, cache_dir=tmp_path)
    models = [ModelProfile("small", 4096, 512), ModelProfile("large", 32768, 2048)]
    configs = main_configs([5, 10]) + ablation_configs()

    async def go():
        indexed = await index_sessions(sessions, embedder, ("header", "plain"))
        return indexed, await evaluate(indexed, configs, models, embedder)
    indexed, rows = asyncio.run(go())

    questions = sum(len(s.questions) for s in sessions)
    assert len(rows) == questions * len(configs)
    assert all(ids for s in indexed for ids in s.gold_ids.values())       # every gold maps to a chunk
    v0 = [r for r in rows if r["config"] == "V0"]
    assert all(r["retrieval"] is None and r["context"]["small"]["fragments"] == 0 for r in v0)
    v2 = [r for r in rows if r["config"] == "V2@10"]
    assert all(0 <= r["retrieval"]["rr"] <= 1 for r in v2)
    assert any(r["context"]["small"]["via"] == "fragment" for r in v2)

    # a second run embeds nothing new: everything is in the cache
    before = fake.calls
    asyncio.run(index_sessions(sessions, CachedEmbedder(fake, cache_dir=tmp_path), ("header", "plain")))
    assert fake.calls == before

    meta = {"sessions": 2, "turns": 24, "questions": questions, "chunks": 1, "chunker_version": "1",
            "embedding_model": "bag-of-words", "models": [m.__dict__ for m in models]}
    text = write_report(rows, meta, tmp_path / "report.md")
    assert "| V2@10 |" in text and "| V3 | — |" in text and "## Ablations" in text
