"""The evaluation harness (scripts/rag_eval/): the dataset is valid, gold quotes map to chunks, and the
harness runs end to end with a fake embedding model (no Ollama)."""
import asyncio
import hashlib
import json
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


# ------------------------------------------------------------------ judge_retrieval.py (no network)
def test_parse_grades_tolerates_thinking_fences_and_prose():
    from judge_retrieval import parse_grades
    reply = ('<think>maybe {"grades": [{"id": 1, "grade": 0}]}</think>\nSure:\n```json\n'
             '{"grades": [{"id": 1, "grade": 2, "reason": "the budget"}, {"id": "2", "grade": 1}, '
             '{"id": 3, "grade": 7}, {"grade": 2}]}\n```')
    assert parse_grades(reply) == {1: (2, "the budget"), 2: (1, "")}
    assert parse_grades("no json at all") == {}


def test_ndcg():
    from judge_retrieval import ndcg
    # dcg([2, 0, 1]) = 3/1 + 0 + 1/2 = 3.5, ideal dcg([2, 1, 0]) = 3 + 1/log2(3) = 3.631
    assert ndcg([2, 0, 1], [2, 1, 0], 3) == pytest.approx(0.9639, abs=1e-3)
    assert ndcg([2, 1], [2, 1], 2) == 1.0
    assert ndcg([0, 0], [0, 0], 2) == 0.0


def test_judge_retrieval_end_to_end(tmp_path):
    import re

    from judge_retrieval import GradeCache, judge_retrieval
    from judge_retrieval import write_report as write_judge_report
    sessions = [s for s in SESSIONS if s.id.startswith(("s01", "s03"))]
    embedder = CachedEmbedder(BagOfWordsEmbedder(), cache_dir=tmp_path)
    calls = []

    async def fake_judge(prompt):                    # grades every excerpt 1, the first one 2
        calls.append(prompt)
        ids = [int(i) for i in re.findall(r"^\[id (\d+)\]", prompt, re.M)]
        return json.dumps({"grades": [{"id": i, "grade": 2 if i == 1 else 1, "reason": "x"} for i in ids]})

    async def go():
        indexed = await index_sessions(sessions, embedder)
        return await judge_retrieval(indexed, main_configs([5, 10]), embedder, fake_judge,
                                     GradeCache("fake", cache_dir=tmp_path), batch=4)
    metrics, judged = asyncio.run(go())

    questions = sum(len(s.questions) for s in sessions)
    assert len(metrics) == questions * 4                   # V1@5, V1@10, V2@5, V2@10: V0 is skipped
    assert all(m["unjudged"] == 0 for m in metrics)
    assert all(m["judge_hit"] for m in metrics if m["config"].endswith("@10"))
    assert any(j["gold"] for j in judged)

    # a second run asks the judge nothing: every grade is cached
    before = len(calls)
    asyncio.run(go())
    assert len(calls) == before

    text = write_judge_report(metrics, judged, {"sessions": 2, "questions": questions, "judged": len(judged),
                                                "judge_model": "fake"}, tmp_path / "report.md")
    assert "| V2@10 |" in text and "## Judge vs gold labels" in text


# ------------------------------------------------------------------ answer_eval.py (no network)
class FakeEngine:
    """Stands in for src/engine: answers with the last user message, records every call."""

    def __init__(self):
        self.calls = []

    async def chat(self, model, messages, max_tokens, num_ctx):
        from engine import ChatResult
        self.calls.append((model, max_tokens, num_ctx))
        return ChatResult(text=f"<think>hmm</think>{model}: {messages[-1]['content'][:40]}", prompt_tokens=10,
                          completion_tokens=5)


def test_answer_eval_specs_and_configs():
    from answer_eval import parse_config, parse_model_spec
    assert parse_model_spec("expert").engine is None
    spec = parse_model_spec("ollama:gemma3:1b")
    assert (spec.engine, spec.model, spec.label) == ("ollama", "gemma3:1b", "ollama:gemma3:1b")
    with pytest.raises(ValueError):
        parse_model_spec("gemma3")
    assert parse_config("V0").top_k == 0
    assert (parse_config("V2@10").preset.name, parse_config("V2@10").top_k) == ("V2", 10)
    for bad in ("V1", "V9@5", "V3@10"):
        with pytest.raises(ValueError):
            parse_config(bad)


def test_mcnemar_and_verdicts():
    from answer_eval import mcnemar_p, parse_verdict
    assert mcnemar_p(6, 17) == pytest.approx(0.035, abs=1e-3)
    assert mcnemar_p(0, 0) == 1.0 and mcnemar_p(5, 5) == 1.0
    reply = '<think>{"score": 1}</think> ok ```json\n{"score": 4, "outdated": true, "reason": "old port"}\n```'
    assert parse_verdict(reply) == {"score": 4, "outdated": True, "reason": "old port"}
    with pytest.raises(ValueError):
        parse_verdict('{"score": 9}')


def test_answer_eval_end_to_end(tmp_path):
    from answer_eval import (
        CORRECT,
        JsonlCache,
        answer_all,
        build_tasks,
        excerpts,
        judge_all,
        judge_prompt,
        parse_config,
        parse_model_spec,
        write_report,
    )

    from router.routing_rule import Model
    from shared.config import RetrievalSettings
    sessions = [s for s in SESSIONS if s.id.startswith(("s01", "s03"))]
    catalog = [Model("coder", "coding", "high"), Model("mathy", "math", "high"),
               Model("general", None, "high", is_default=True, context_window=8192, max_output_tokens=256)]
    specs = [parse_model_spec("expert"), parse_model_spec("ollama:tiny")]
    configs = [parse_config(c) for c in ("V0", "V1@10")]
    embedder = CachedEmbedder(BagOfWordsEmbedder(), cache_dir=tmp_path)
    engine = FakeEngine()

    async def go():
        from harness import query_vectors
        indexed = await index_sessions(sessions, embedder)
        vectors = await query_vectors(indexed, embedder)
        tasks = build_tasks(indexed, configs, specs, vectors, RetrievalSettings(), catalog, "high", 2048, 256,
                            limit=4)
        return tasks, await answer_all(tasks, {"ollama": engine}, cache_dir=tmp_path)
    tasks, rows = asyncio.run(go())

    assert len(rows) == 4 * len(configs) * len(specs)                  # --limit 4 questions
    experts = {r["model"] for r in rows if r["set"] == "expert"}
    assert experts == {"coder"}                                         # s01 is a coding session
    assert {(m, n) for m, _, n in engine.calls} == {("coder", 4096), ("tiny", 2048)}
    assert all(r["context"]["tokens"] < r["context_window"] for r in rows)       # each fits its model
    assert all(t.messages[-1]["role"] == "user" for t in tasks)

    # a second run answers from the cache
    before = len(engine.calls)
    asyncio.run(answer_all(tasks, {"ollama": engine}, cache_dir=tmp_path))
    assert len(engine.calls) == before

    s01 = sessions[0]
    q = s01.questions[0]
    assert q.gold[0].quote.split("\n")[0][:30] in excerpts(s01, q.gold)
    prompt = judge_prompt(s01, q.id, "<think>x</think>bcrypt")
    assert "bcrypt" in prompt and "<think>" not in prompt

    async def fake_judge(prompt):                    # right when the answer quotes the question back
        return json.dumps({"score": 5 if "Cum" in prompt.split("Answer to grade:")[1] else 2,
                           "outdated": False, "reason": "x"})

    async def judge():
        return await judge_all(rows, {s.id: s for s in sessions}, fake_judge, JsonlCache(tmp_path / "j.jsonl"))
    judged = asyncio.run(judge())
    assert all(1 <= r["judge"]["score"] <= 5 for r in judged)
    text = write_report(judged, {"questions": 4, "judge_model": "fake"}, tmp_path / "report.md")
    assert "## Main table" in text and "| expert | V1@10 |" in text and "V1@10 vs V0" in text
    assert "## Expert models" in text and CORRECT == 4
