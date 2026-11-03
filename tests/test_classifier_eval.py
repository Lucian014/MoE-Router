"""scripts/classifier_eval/long_prompts.py without the classifiers: the cases it builds and the ablation
configs (the input rules themselves are tested in test_classifier_input.py)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "classifier_eval"))

from long_prompts import REQUESTS, build_cases, configs_from, dataset_pieces  # noqa: E402


def words(text: str) -> int:
    return len(text.split())


def test_cases_cover_every_request_size_and_position():
    cases = build_cases(dataset_pieces(), words, sizes=(100, 400), variants=1)
    assert len(cases) == len(REQUESTS) * (1 + 2 * 2)
    for c in cases:
        if c.position == "alone":
            assert c.size is None and c.prompt in {r.start for r in REQUESTS}
        else:
            assert c.tokens >= c.size                # the document reached its target length
    start = next(c for c in cases if c.position == "start")
    end = next(c for c in cases if c.position == "end" and c.label == start.label)
    req = REQUESTS[int(start.id.split("-")[0][1:])]
    assert start.prompt.startswith(req.start) and end.prompt.endswith(req.end)


def test_documents_carry_none_of_the_datasets_requests():
    pieces = dataset_pieces()
    assert pieces["prose"] and pieces["code"]
    assert all(len(p.split()) >= 40 for p in pieces["prose"])


def test_ablation_configs():
    names = [c.name for c in configs_from(["raw", "edges"], [512, 2048], [60, 100])]
    assert names == ["raw@512", "edges@512", "edges/100@512", "raw@2048", "edges@2048", "edges/100@2048"]
