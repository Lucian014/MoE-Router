"""Ranking the candidates of R2 (docs/rag-planv2.md 4.5, 4.6). Pure functions, no I/O: the router
(and the harness) load the candidates -- every chunk of the session with its distance to the
query and its BM25 terms -- and get back the pool, best first.

    dense   cosine distance computed by Postgres (pgvector), smallest first
    BM25    over the candidates' `terms` (k1=1.2, b=0.75), IDF on the whole indexed corpus
            (TermStats) when given, else on the session's chunks. On the corpus, words that are
            everywhere -- function words in any language -- weigh almost nothing, so no word
            list is needed; on one session's 20-100 chunks, "did" or "pentru" can look rare
    RRF     score = sum over the methods of 1 / (rrf_k + rank), each method's top `candidates`
    dedup   same content_hash ("hash"), or Jaccard on the term sets >= jaccard ("near"):
            only the NEWEST version stays, at the best position of its group, so an old version
            of rewritten code never reaches the model in place of the new one
    pool    the first `pool` after dedup; the router cuts it to top-k once the model, and so
            the turns included whole, are known (assembly.py)
    gate    (gate_similarity > 0) a request that has nothing to do with the conversation ("write
            me a haiku") gets no fragments at all: when even the closest chunk's dense similarity
            (1 - cosine distance) is below the gate, the pool is empty. Dense only: BM25 and RRF
            scores depend on the query and can't be compared to a fixed threshold
    floor   (min_similarity > 0) each fragment of the pool must reach it on its own; the chunks
            brought by `next` are exempt (the one that answers doesn't look like the question)
    next    (neighbor_turns > 0) multi-hop: "why did X fail?" is often answered in the turn AFTER
            the one that names X (a log pasted, "that was it"), which shares no words with the
            question. Each of the first `neighbor_from_top` of the pool brings the first chunk of
            the answer of each of its next `neighbor_turns` turns right after it: where the
            diagnosis usually is. Not the chunk closest to the query: the one that answers is,
            by construction, the one that doesn't look like the question

Ties (equal score) go to the newer turn, then to the earlier chunk of the turn.
"""
import math
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
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
    source: str               # prompt | response


@dataclass(frozen=True)
class TermStats:
    """Document frequencies over a corpus (every indexed chunk): `docs` chunks in all, `df[t]` of
    them contain t. Only the query's terms are needed."""

    docs: int
    df: Mapping[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class RankParams:
    candidates: int = 50          # per method, before fusion (RAG_CANDIDATES)
    pool: int = 30                # after dedup (RAG_POOL)
    rrf_k: int = 60
    dedup: str = "near"           # off | hash | near (RAG_DEDUP)
    jaccard: float = 0.8          # "near" threshold (RAG_DEDUP_JACCARD)
    neighbor_turns: int = 0       # next turns a top chunk brings along, 0 = off (RAG_NEIGHBOR_TURNS)
    neighbor_from_top: int = 3    # how many of the first chunks do (RAG_NEIGHBOR_FROM_TOP)
    gate_similarity: float = 0.0  # no fragments when the best chunk is below it, 0 = off (RAG_GATE_SIMILARITY)
    min_similarity: float = 0.0   # each fragment at least this similar, 0 = off (RAG_MIN_SIMILARITY)

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
                k1: float = BM25_K1, b: float = BM25_B, corpus: TermStats | None = None) -> list[float]:
    """Okapi BM25 of every doc for the query; each distinct query term counts once (a pasted
    traceback repeats words, which shouldn't multiply their weight). `corpus`: the IDF comes from
    it instead of from `docs` (the docs are part of the corpus, so their counts are a floor)."""
    n = len(docs)
    if not n or not query_terms:
        return [0.0] * n
    lengths = [len(d) for d in docs]
    avgdl = (sum(lengths) / n) or 1.0
    df = Counter(t for d in docs for t in set(d))
    query = [t for t in dict.fromkeys(query_terms) if t in df]
    total = max(corpus.docs, n) if corpus else n
    corpus_df = corpus.df if corpus else {}
    idf = {}
    for t in query:
        d = max(df[t], corpus_df.get(t, 0))
        idf[t] = math.log(1 + (total - d + 0.5) / (d + 0.5))
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


def bm25_order(candidates: Sequence[Candidate], query_terms: Sequence[str],
               corpus: TermStats | None = None) -> list[Candidate]:
    scores = bm25_scores(query_terms, [c.terms for c in candidates], corpus=corpus)
    scored = [(s, c) for s, c in zip(scores, candidates) if s > 0]
    return [c for s, c in sorted(scored, key=lambda x: (-x[0], *_newest_first(x[1])))]


def rank(candidates: Sequence[Candidate], query_terms: Sequence[str], preset: Preset,
         params: RankParams = RankParams(), corpus: TermStats | None = None) -> list[Ranked]:
    """The pool: the candidates of the preset's methods, fused, deduplicated, best first.
    `corpus`: BM25's document frequencies over every indexed chunk (else the session's)."""
    lists: dict[str, list[Candidate]] = {}
    if preset.dense:
        lists["dense"] = dense_order(candidates)[:params.candidates]
    if preset.bm25:
        lists["bm25"] = bm25_order(candidates, query_terms, corpus)[:params.candidates]

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
    pool = deduplicate(ranked, params.dedup, params.jaccard)[:params.pool]
    pool = relevant(pool, candidates, params.gate_similarity, params.min_similarity)
    return add_neighbors(pool, candidates, params.neighbor_turns, params.neighbor_from_top)


def similarity(c: Candidate) -> float | None:
    """Cosine similarity to the query (pgvector's cosine distance is 1 - similarity)."""
    return None if c.distance is None else 1.0 - c.distance


def relevant(pool: Sequence[Ranked], candidates: Sequence[Candidate], gate: float = 0.0,
             floor: float = 0.0) -> list[Ranked]:
    """The gate (nothing when the closest chunk of the session is below `gate`), then the floor (each
    chunk at least `floor`). Without dense distances (dense search off) neither applies."""
    sims = [s for c in candidates if (s := similarity(c)) is not None]
    if not sims:
        return list(pool)
    if gate > 0 and max(sims) < gate:
        return []
    if floor > 0:
        return [r for r in pool if (s := similarity(r.candidate)) is None or s >= floor]
    return list(pool)


def add_neighbors(pool: Sequence[Ranked], candidates: Sequence[Candidate], turns: int = 1,
                  from_top: int = 3) -> list[Ranked]:
    """After each of the first `from_top` chunks of the pool, the first chunk of the answer of each of
    its next `turns` turns (of the prompt, for a turn whose answer has no chunk), with the score of
    the chunk that brought it. A chunk already placed stays where it was first put."""
    if turns <= 0 or not pool:
        return list(pool)
    by_turn: dict[int, list[Candidate]] = defaultdict(list)
    for c in candidates:
        by_turn[c.turn].append(c)

    def answer_start(turn: int) -> Candidate | None:
        chunks = by_turn.get(turn)
        return min(chunks, key=lambda c: (c.source != "response", c.ord)) if chunks else None

    out: list[Ranked] = []
    placed: set[int] = set()
    for i, r in enumerate(pool):
        if r.id in placed:
            continue
        out.append(r)
        placed.add(r.id)
        if i >= from_top:
            continue
        for d in range(1, turns + 1):
            n = answer_start(r.turn + d)
            if n is not None and n.id not in placed:
                out.append(Ranked(n, r.score))
                placed.add(n.id)
    return out


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
