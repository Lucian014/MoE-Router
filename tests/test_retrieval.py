"""Ranking (src/shared/memory/retrieval.py) and the presets (src/shared/memory/presets.py)."""
from dataclasses import dataclass, field

import pytest

from shared.memory.analyzer import analyze
from shared.memory.presets import PRESETS, get_preset
from shared.memory.retrieval import RankParams, bm25_scores, deduplicate, jaccard, rank

V1, V2 = PRESETS["V1"], PRESETS["V2"]


@dataclass(frozen=True)
class C:
    id: int
    turn: int
    ord: int = 0
    distance: float | None = 0.5
    terms: list[str] = field(default_factory=list)
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            object.__setattr__(self, "content_hash", f"h{self.id}")


def ids(ranked):
    return [r.id for r in ranked]


# ------------------------------------------------------------------ presets
def test_presets():
    assert [p for p in PRESETS] == ["V0", "V1", "V2", "V3", "V4", "V5"]
    assert not PRESETS["V0"].retrieves
    assert (V1.dense, V1.bm25, V2.dense, V2.bm25) == (True, False, True, True)
    assert [p.name for p in PRESETS.values() if p.online] == ["V0", "V1", "V2"]
    with pytest.raises(ValueError, match="unknown RAG variant"):
        get_preset("V9")


# ------------------------------------------------------------------ BM25
def test_bm25_prefers_rare_terms_and_ignores_missing_ones():
    docs = [["jwt", "token", "expir"], ["token", "list"], ["sort", "list"]]
    scores = bm25_scores(["jwt", "token", "absent"], docs)
    assert scores[0] > scores[1] > 0 and scores[2] == 0


def test_bm25_counts_a_repeated_query_term_once():
    docs = [["error", "x"], ["y", "z"]]
    assert bm25_scores(["error"] * 20, docs) == bm25_scores(["error"], docs)


def test_bm25_on_nothing():
    assert bm25_scores(["a"], []) == [] and bm25_scores([], [["a"]]) == [0.0]


# ------------------------------------------------------------------ ranking
def test_dense_orders_by_distance_and_skips_chunks_without_one():
    cands = [C(1, 1, distance=0.4), C(2, 1, ord=1, distance=0.1), C(3, 2, distance=None)]
    assert ids(rank(cands, [], V1)) == [2, 1]


def test_hybrid_fuses_with_rrf():
    # 1: best dense, absent from BM25. 2: second in both -> wins the fusion
    cands = [C(1, 1, distance=0.1, terms=["other"]),
             C(2, 2, distance=0.2, terms=["create_access_token", "jwt"]),
             C(3, 3, distance=0.9, terms=["create_access_token", "jwt", "jwt"]),
             C(4, 4, distance=0.5, terms=["more"])]
    ranked = rank(cands, analyze("create_access_token jwt"), V2, RankParams(dedup="off"))
    assert ids(ranked) == [2, 3, 1, 4]       # 3: first in BM25 but last in dense
    top = ranked[0]
    assert (top.dense_rank, top.bm25_rank) == (2, 2)
    assert top.score == pytest.approx(2 / 62)


def test_equal_scores_go_to_the_newer_turn():
    cands = [C(1, turn=1, distance=0.3), C(2, turn=5, distance=0.3)]
    assert ids(rank(cands, [], V1)) == [2, 1]


def test_pool_and_per_method_limits():
    cands = [C(i, i, distance=i / 100) for i in range(1, 21)]
    assert ids(rank(cands, [], V1, RankParams(candidates=5, pool=3))) == [1, 2, 3]
    assert len(rank(cands, [], V1, RankParams(candidates=5, pool=30))) == 5


def test_v1_ignores_bm25_and_bm25_alone_needs_the_terms():
    cands = [C(1, 1, distance=0.9, terms=["jwt"]), C(2, 2, distance=0.1, terms=["sort"])]
    assert ids(rank(cands, ["jwt"], V1)) == [2, 1]


# ------------------------------------------------------------------ dedup
CODE_V1 = """def load_users(path):
    with open(path, encoding="utf-8") as f:
        rows = json.load(f)
    users = [User(**row) for row in rows if row.get("active")]
    return sorted(users, key=lambda u: u.created_at)
"""
CODE_V2 = CODE_V1.replace("key=lambda u: u.created_at)", "key=lambda u: u.created_at, reverse=True)")


def test_same_code_pasted_again_keeps_only_the_newest_turn_at_the_best_position():
    cands = [C(1, turn=2, distance=0.1, terms=analyze(CODE_V1), content_hash="same"),
             C(2, turn=3, distance=0.5, terms=["other"]),
             C(3, turn=6, distance=0.6, terms=analyze(CODE_V1), content_hash="same")]
    ranked = rank(cands, [], V1, RankParams(dedup="hash"))
    assert ids(ranked) == [3, 2]                     # turn 6 took turn 2's first place


def test_rewritten_code_is_a_near_duplicate():
    a, b = set(analyze(CODE_V1)), set(analyze(CODE_V2))
    assert jaccard(a, b) >= 0.8
    cands = [C(1, turn=2, distance=0.1, terms=analyze(CODE_V1)),
             C(2, turn=7, distance=0.3, terms=analyze(CODE_V2))]
    assert ids(rank(cands, [], V1, RankParams(dedup="near"))) == [2]
    assert ids(rank(cands, [], V1, RankParams(dedup="hash"))) == [1, 2]
    assert ids(rank(cands, [], V1, RankParams(dedup="off"))) == [1, 2]


def test_different_chunks_with_common_terms_both_stay():
    first = analyze("Funcția create_access_token semnează tokenul JWT cu cheia secretă din config.")
    second = analyze("Pentru refresh, create_access_token primește o expirare mai lungă, de 7 zile.")
    cands = [C(1, turn=2, distance=0.1, terms=first), C(2, turn=5, distance=0.2, terms=second)]
    assert ids(rank(cands, [], V1)) == [1, 2]


def test_near_dedup_compares_different_turns_only():
    same_terms = ["a", "b", "c"]
    ranked = [r for r in rank([C(1, 3, 0, 0.1, same_terms), C(2, 3, 1, 0.2, same_terms)], [], V1)]
    assert ids(ranked) == [1, 2]
    assert ids(deduplicate(ranked, "near")) == [1, 2]


def test_unknown_dedup_mode_is_rejected():
    with pytest.raises(ValueError):
        RankParams(dedup="fuzzy")
