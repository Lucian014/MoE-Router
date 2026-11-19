"""Ranking (src/shared/memory/retrieval.py) and the presets (src/shared/memory/presets.py)."""
from dataclasses import dataclass, field

import pytest

from shared.memory.analyzer import analyze
from shared.memory.presets import PRESETS, get_preset
from shared.memory.retrieval import (
    RankParams,
    TermStats,
    add_neighbors,
    bm25_scores,
    deduplicate,
    jaccard,
    rank,
    relevant,
)

V1, V2 = PRESETS["V1"], PRESETS["V2"]


@dataclass(frozen=True)
class C:
    id: int
    turn: int
    ord: int = 0
    distance: float | None = 0.5
    terms: list[str] = field(default_factory=list)
    content_hash: str = ""
    source: str = "response"

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


def test_corpus_idf_weighs_common_words_down():
    # in the session alone, "did" looks as rare as "redis_url"; over the corpus it is everywhere
    docs = [["did", "we", "go"], ["redis_url", "redis", "auth"], ["other", "text"]]
    query = ["did", "redis_url"]
    alone = bm25_scores(query, docs)
    assert alone[0] == pytest.approx(alone[1], rel=0.2)
    corpus = TermStats(docs=10_000, df={"did": 6_000, "we": 7_000, "redis_url": 3})
    weighted = bm25_scores(query, docs, corpus=corpus)
    assert weighted[1] > 5 * weighted[0] > 0


def test_corpus_counts_are_never_below_the_session():
    # the session's chunks are part of the corpus: a stale or partial corpus can't make a term rarer
    docs = [["a", "x"], ["a", "y"], ["a", "z"]]
    tiny = TermStats(docs=1, df={})
    assert bm25_scores(["a", "x"], docs, corpus=tiny) == pytest.approx(bm25_scores(["a", "x"], docs))


# ------------------------------------------------------------------ next-turn neighbors (multi-hop)
def test_neighbors_are_off_by_default():
    cands = [C(1, turn=4, distance=0.1), C(2, turn=5, distance=0.9)]
    assert ids(rank(cands, [], V1)) == [1, 2]                        # only the dense order
    assert ids(rank(cands, [], V1, RankParams(pool=1))) == [1]


def test_a_top_chunk_brings_the_start_of_the_next_answer_right_after_it():
    # turn 4 names the problem (close to the question); turn 5: a pasted log, then the diagnosis, then
    # more -- the diagnosis follows 4 even though another chunk of 5 is closer to the question
    cands = [C(1, turn=4, distance=0.1), C(2, turn=5, ord=0, distance=0.5, source="prompt"),
             C(3, turn=5, ord=1, distance=0.8), C(4, turn=5, ord=2, distance=0.6),
             C(5, turn=9, distance=0.2), C(6, turn=10, distance=0.95)]
    pool = rank(cands, [], V1, RankParams(pool=2, neighbor_turns=1))
    assert ids(pool) == [1, 3, 5, 6]
    assert pool[1].score == pool[0].score                              # the score of the chunk that brought it


def test_only_the_first_chunks_bring_neighbors_and_nothing_is_placed_twice():
    cands = [C(1, turn=4, distance=0.1), C(2, turn=5, distance=0.2), C(3, turn=6, distance=0.3),
             C(4, turn=8, distance=0.4), C(5, turn=9, distance=0.9)]
    pool = rank(cands, [], V1, RankParams(pool=4, neighbor_turns=1, neighbor_from_top=2))
    assert ids(pool) == [1, 2, 3, 4]          # 1 brings 2 (already next), 2 brings 3; 4 is past the top 2


def test_neighbors_of_several_turns_of_the_last_turn_and_of_a_prompt_only_turn():
    cands = [C(1, turn=4, distance=0.1), C(2, turn=5, distance=0.9), C(3, turn=6, distance=0.9),
             C(4, turn=7, distance=0.2), C(5, turn=8, ord=1, source="prompt"), C(6, turn=8, ord=0, source="prompt")]
    assert ids(add_neighbors(rank(cands[:4], [], V1, RankParams(pool=1)), cands, turns=2)) == [1, 2, 3]
    assert ids(add_neighbors(rank([cands[3]], [], V1), cands, turns=1)) == [4, 6]
    assert ids(add_neighbors(rank([cands[4]], [], V1), cands[:5], turns=1)) == [5]   # no turn 9


# ------------------------------------------------------------------ relevance thresholds
def test_thresholds_are_off_by_default():
    cands = [C(1, turn=1, distance=0.9), C(2, turn=2, distance=0.95)]       # similarity 0.10 / 0.05
    assert ids(rank(cands, [], V1)) == [1, 2]


def test_the_gate_empties_the_pool_when_nothing_in_the_session_is_close_enough():
    cands = [C(1, turn=1, distance=0.6), C(2, turn=2, distance=0.7)]        # best similarity 0.40
    assert ids(rank(cands, [], V1, RankParams(gate_similarity=0.45))) == []
    assert ids(rank(cands, [], V1, RankParams(gate_similarity=0.40))) == [1, 2]   # passes: everything stays


def test_the_floor_drops_weak_fragments_but_not_the_neighbors_they_bring():
    cands = [C(1, turn=4, distance=0.3), C(2, turn=5, distance=0.9), C(3, turn=9, distance=0.7)]
    assert ids(rank(cands, [], V1, RankParams(min_similarity=0.5))) == [1]
    assert ids(rank(cands, [], V1, RankParams(min_similarity=0.5, neighbor_turns=1))) == [1, 2]


def test_thresholds_need_dense_distances():
    cands = [C(1, turn=1, distance=None, terms=["x"])]
    assert ids(relevant(rank(cands, ["x"], PRESETS["V2"]), cands, gate=0.9, floor=0.9)) == [1]
