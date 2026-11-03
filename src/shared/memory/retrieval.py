"""Ranking the candidates of R2 (docs/rag-planv2.md 4.5, 4.6). Pure functions, no I/O: the router
(and the harness) load the candidates -- every chunk of the session with its distance to the
query and its BM25 terms -- and get back the pool, best first.

    dense   cosine distance computed by Postgres (pgvector), smallest first
    BM25    over the candidates' `terms`, IDF on the session's chunks (k1=1.2, b=0.75)
    RRF     score = sum over the methods of 1 / (rrf_k + rank), each method's top `candidates`
    dedup   same content_hash ("hash"), or Jaccard on the term sets >= jaccard ("near"):
            only the NEWEST version stays, at the best position of its group, so an old version
            of rewritten code never reaches the model in place of the new one
    pool    the first `pool` after dedup; the router cuts it to top-k once the model, and so
            the turns included whole, are known (assembly.py)

Ties (equal score) go to the newer turn, then to the earlier chunk of the turn.
"""
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Protocol

from shared.memory.presets import Preset

DEDUP_MODES = ("off", "hash", "near")
BM25_K1 = 1.2
BM25_B = 0.75


class Candidate(Protocol):
    """database.ChunkCandidate, or the harness's in-memory equivalent."""

    id: int
    turn: int
    ord: int
    content_hash: str
    terms: list[str]
    distance: float | None


@dataclass(frozen=True)
class RankParams:
    candidates: int = 50          # per method, before fusion (RAG_CANDIDATES)
    pool: int = 30                # after dedup (RAG_POOL)
    rrf_k: int = 60
    dedup: str = "near"           # off | hash | near (RAG_DEDUP)
    jaccard: float = 0.8          # "near" threshold (RAG_DEDUP_JACCARD)

    def __post_init__(self):
        if self.dedup not in DEDUP_MODES:
            raise ValueError(f"dedup must be one of {DEDUP_MODES}, got {self.dedup!r}")


@dataclass(frozen=True)
class Ranked:
    candidate: Candidate
    score: float
    dense_rank: int | None = None     # 1-based, None = not in that method's top
    bm25_rank: int | None = None

    @property
    def id(self) -> int:
        return self.candidate.id

    @property
    def turn(self) -> int:
        return self.candidate.turn


def _newest_first(c: Candidate) -> tuple[int, int]:
    return -c.turn, c.ord


def bm25_scores(query_terms: Sequence[str], docs: Sequence[Sequence[str]],
                k1: float = BM25_K1, b: float = BM25_B) -> list[float]:
    """Okapi BM25 of every doc for the query; each distinct query term counts once (a pasted
    traceback repeats words, which shouldn't multiply their weight)."""
    n = len(docs)
    if not n or not query_terms:
        return [0.0] * n
    lengths = [len(d) for d in docs]
    avgdl = (sum(lengths) / n) or 1.0
    df = Counter(t for d in docs for t in set(d))
    query = [t for t in dict.fromkeys(query_terms) if t in df]
    idf = {t: math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5)) for t in query}
    scores = []
    for doc, length in zip(docs, lengths):
        tf = Counter(doc)
        s = 0.0
        for t in query:
            f = tf.get(t, 0)
            if f:
                s += idf[t] * f * (k1 + 1) / (f + k1 * (1 - b + b * length / avgdl))
        scores.append(s)
    return scores


def dense_order(candidates: Sequence[Candidate]) -> list[Candidate]:
    return sorted((c for c in candidates if c.distance is not None),
                  key=lambda c: (c.distance, *_newest_first(c)))


def bm25_order(candidates: Sequence[Candidate], query_terms: Sequence[str]) -> list[Candidate]:
    scores = bm25_scores(query_terms, [c.terms for c in candidates])
    scored = [(s, c) for s, c in zip(scores, candidates) if s > 0]
    return [c for s, c in sorted(scored, key=lambda x: (-x[0], *_newest_first(x[1])))]


def rank(candidates: Sequence[Candidate], query_terms: Sequence[str], preset: Preset,
         params: RankParams = RankParams()) -> list[Ranked]:
    """The pool: the candidates of the preset's methods, fused, deduplicated, best first."""
    lists: dict[str, list[Candidate]] = {}
    if preset.dense:
        lists["dense"] = dense_order(candidates)[:params.candidates]
    if preset.bm25:
        lists["bm25"] = bm25_order(candidates, query_terms)[:params.candidates]

    by_id: dict[int, Candidate] = {}
    score: dict[int, float] = {}
    ranks: dict[str, dict[int, int]] = {}
    for method, ordered in lists.items():
        ranks[method] = {}
        for r, c in enumerate(ordered, start=1):
            by_id[c.id] = c
            score[c.id] = score.get(c.id, 0.0) + 1.0 / (params.rrf_k + r)
            ranks[method][c.id] = r
    fused = sorted(by_id.values(), key=lambda c: (-score[c.id], *_newest_first(c)))
    ranked = [Ranked(c, score[c.id], ranks.get("dense", {}).get(c.id), ranks.get("bm25", {}).get(c.id))
              for c in fused]
    return deduplicate(ranked, params.dedup, params.jaccard)[:params.pool]


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def deduplicate(ranked: Sequence[Ranked], mode: str = "near", threshold: float = 0.8) -> list[Ranked]:
    """Walk best first; a candidate that duplicates one already kept replaces it if it is newer
    (taking its position and score), else it is dropped. "near" compares chunks of different turns
    only: two chunks of one turn share the header terms, not the content."""
    if mode == "off":
        return list(ranked)
    kept: list[Ranked] = []
    term_sets: list[set[str]] = []
    for r in ranked:
        terms = set(r.candidate.terms)
        dup = next((i for i, k in enumerate(kept)
                    if k.candidate.content_hash == r.candidate.content_hash
                    or (mode == "near" and k.turn != r.turn and jaccard(term_sets[i], terms) >= threshold)),
                   None)
        if dup is None:
            kept.append(r)
            term_sets.append(terms)
        elif (r.candidate.turn, -r.candidate.ord) > (kept[dup].candidate.turn, -kept[dup].candidate.ord):
            kept[dup] = replace(r, score=kept[dup].score)
            term_sets[dup] = terms
    return kept
