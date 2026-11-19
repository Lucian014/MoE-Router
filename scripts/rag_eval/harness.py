"""
The offline harness (docs/rag-planv2.md 6): the same functions the memory service and the router
run (src/shared/memory/: header, chunking, analyzer, retrieval, assembly), without RabbitMQ,
Postgres or a cluster. Only R2 differs: the cosine distance is computed here instead of by
pgvector (same formula; the SQL itself is covered by tests/test_database_memory.py).

    index_sessions()   every turn -> header, chunks, terms, embeddings (cached on disk)
    evaluate()         per config x question: the pool, retrieval metrics, and per model
                       profile whether the gold reached the model (whole turn or fragment)
"""
import hashlib
import json
import math
import re
import sys
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dataset import Gold, Question, Session, gold_ords
from shared.config import RetrievalSettings
from shared.memory.analyzer import analyze
from shared.memory.assembly import AssembledContext, Budget, Fragment, TurnText, build_messages
from shared.memory.chunking import chunk_turn, embedding_input, terms_input
from shared.memory.embedding import EmbeddingClient
from shared.memory.header import DENSE_QUERY_MAX_TOKENS, build_dense_query, extract_question, tail_tokens
from shared.memory.presets import PRESETS, Preset
from shared.memory.retrieval import RankParams, TermStats, rank
from shared.prompts import SYSTEM_PROMPT

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
EMBED_BATCH = 32


# ------------------------------------------------------------------ embeddings, cached
class CachedEmbedder:
    """Every text is embedded once per model, ever: the vectors live in .cache/<model>.jsonl."""

    def __init__(self, client: EmbeddingClient, cache_dir: Path | None = CACHE_DIR):
        self.client, self.model = client, client.model
        self._path = cache_dir / f"{self.model.replace('/', '_').replace(':', '_')}.jsonl" if cache_dir else None
        self._vectors: dict[str, list[float]] = {}
        if self._path and self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self._vectors[row["k"]] = row["v"]

    @staticmethod
    def key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        missing = list(dict.fromkeys(t for t in texts if self.key(t) not in self._vectors))
        for i in range(0, len(missing), EMBED_BATCH):
            batch = missing[i:i + EMBED_BATCH]
            vectors = await self.client.embed(batch)
            new = {self.key(t): v for t, v in zip(batch, vectors)}
            self._vectors.update(new)
            if self._path:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as f:
                    for k, v in new.items():
                        f.write(json.dumps({"k": k, "v": v}) + "\n")
        return [self._vectors[self.key(t)] for t in texts]


def _unit(v: Sequence[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in v)) or 1.0
    return [x / norm for x in v]


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """Both already unit length."""
    return 1.0 - sum(x * y for x, y in zip(a, b))


# ------------------------------------------------------------------ the index
@dataclass
class IndexedChunk:
    id: int
    turn: int
    ord: int
    source: str
    kind: str
    content: str
    tokens: int
    content_hash: str
    header: str
    header_from_answer: bool
    terms_by: dict[str, list[str]]           # "header" (as indexed) | "plain" (ablation: no header)
    vector_by: dict[str, list[float]]


@dataclass
class IndexedSession:
    session: Session
    chunks: list[IndexedChunk]
    gold_ids: dict[str, set[int]] = field(default_factory=dict)        # question id -> chunk ids
    outdated_ids: dict[str, set[int]] = field(default_factory=dict)
    gold_kinds: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class Candidate:
    """retrieval.Candidate: what R2 returns."""
    id: int
    turn: int
    ord: int
    content_hash: str
    terms: list[str]
    distance: float | None
    source: str = "response"


async def index_sessions(sessions: list[Session], embedder: CachedEmbedder,
                         representations: Sequence[str] = ("header",)) -> list[IndexedSession]:
    """Index every turn the way memory/indexer.py does; `representations` adds "plain" (chunks
    embedded without the header) for the header ablation."""
    out, next_id = [], 1
    for s in sessions:
        chunks: list[IndexedChunk] = []
        ords_by_turn: dict[int, dict[int, int]] = {}
        drafts_by_turn = {}
        for n, t in enumerate(s.turns, start=1):
            header, from_answer = extract_question(t.user, t.assistant)
            drafts = chunk_turn(t.user, t.assistant)
            drafts_by_turn[n] = drafts
            ords_by_turn[n] = {}
            for d in drafts:
                chunk = IndexedChunk(
                    id=next_id, turn=n, ord=d.ord, source=d.source, kind=d.kind, content=d.content,
                    tokens=d.tokens, content_hash=d.content_hash, header=header,
                    header_from_answer=from_answer, terms_by={}, vector_by={})
                chunk.terms_by["header"] = analyze(terms_input(header, d.content))
                chunk.terms_by["plain"] = analyze(d.content)
                ords_by_turn[n][d.ord] = next_id
                chunks.append(chunk)
                next_id += 1
        for rep in representations:
            texts = [embedding_input(c.header, c.content) if rep == "header" else c.content for c in chunks]
            for c, v in zip(chunks, await embedder.embed(texts)):
                c.vector_by[rep] = _unit(v)

        indexed = IndexedSession(s, chunks)
        kinds = {c.id: c.kind for c in chunks}

        def ids_of(golds: list[Gold]) -> set[int]:
            return {ords_by_turn[g.turn][o] for g in golds
                    for o in gold_ords(s.turns[g.turn - 1], drafts_by_turn[g.turn], g)}

        for q in s.questions:
            indexed.gold_ids[q.id] = ids_of(q.gold)
            indexed.outdated_ids[q.id] = ids_of(q.outdated)
            indexed.gold_kinds[q.id] = sorted({kinds[i] for i in indexed.gold_ids[q.id]})
        out.append(indexed)
    return out


# ------------------------------------------------------------------ configs
@dataclass(frozen=True)
class ModelProfile:
    name: str
    context_window: int
    max_output_tokens: int


@dataclass(frozen=True)
class Config:
    name: str
    preset: Preset
    top_k: int
    dedup: str = "near"
    max_per_turn: int = 0
    recent_max_share: float = 0.6
    dense_query: str = "extracted"       # extracted (build_dense_query) | full (the whole prompt)
    representation: str = "header"       # header | plain (chunks embedded without the header)
    idf: str = "corpus"                  # corpus (every indexed chunk, as in production) | session
    neighbors: int = 0                   # RAG_NEIGHBOR_TURNS
    neighbor_from_top: int = 3           # RAG_NEIGHBOR_FROM_TOP
    gate: float = 0.0                    # RAG_GATE_SIMILARITY
    min_sim: float = 0.0                 # RAG_MIN_SIMILARITY

    def describe(self) -> dict:
        return {"config": self.name, "preset": self.preset.name, "top_k": self.top_k, "dedup": self.dedup,
                "max_per_turn": self.max_per_turn, "recent_max_share": self.recent_max_share,
                "dense_query": self.dense_query, "representation": self.representation, "idf": self.idf,
                "neighbors": self.neighbors, "neighbor_from_top": self.neighbor_from_top,
                "gate": self.gate, "min_sim": self.min_sim}


def config_name(preset: str, k: int, neighbors: int = 0, from_top: int = 3, per_turn: int = 0,
                gate: float = 0.0, min_sim: float = 0.0) -> str:
    """V2@10, V2@10+n1 (neighbors, from the top 3), V2@10+n1t5, V2@10+pt2 (max chunks per turn),
    V2@10+n1+g0.45 (gate similarity), V2@10+n1+s0.40 (min similarity per fragment)."""
    name = f"{preset}@{k}"
    if neighbors:
        name += f"+n{neighbors}" + (f"t{from_top}" if from_top != 3 else "")
    if per_turn:
        name += f"+pt{per_turn}"
    if gate:
        name += f"+g{gate:.2f}"
    if min_sim:
        name += f"+s{min_sim:.2f}"
    return name


def parse_config_name(name: str) -> Config:
    """The inverse of config_name (plus "V0")."""
    m = re.fullmatch(r"(V\d)(?:@(\d+))?((?:\+(?:n\d+(?:t\d+)?|pt\d+|[gs][01]\.\d+))*)", name)
    if not m or m.group(1) not in PRESETS:
        raise ValueError(f"config {name!r}: expected V0 or <preset>@<k>[+n<turns>[t<from top>]][+pt<per turn>]"
                         f"[+g<gate similarity>][+s<min similarity>]")
    p = PRESETS[m.group(1)]
    if p.retrieves and not m.group(2):
        raise ValueError(f"config {name!r}: a retrieving preset needs its top-k, e.g. {m.group(1)}@10")
    opts = dict(neighbors=0, neighbor_from_top=3, max_per_turn=0, gate=0.0, min_sim=0.0)
    for part in filter(None, m.group(3).split("+")):
        if part.startswith("pt"):
            opts["max_per_turn"] = int(part[2:])
        elif part[0] in "gs":
            opts["gate" if part[0] == "g" else "min_sim"] = float(part[1:])
        else:
            n, _, t = part[1:].partition("t")
            opts["neighbors"], opts["neighbor_from_top"] = int(n), int(t or 3)
    return Config(name, p, int(m.group(2)) if p.retrieves else 0, **opts)


def main_configs(top_ks: Sequence[int], neighbors: Sequence[int] = (0,), from_tops: Sequence[int] = (3,),
                 per_turns: Sequence[int] = (0,)) -> list[Config]:
    configs = [Config("V0", PRESETS["V0"], top_k=0)]
    for name in ("V1", "V2"):
        for k in top_ks:
            for pt in per_turns:
                for n in neighbors:
                    for t in (from_tops if n else (3,)):
                        configs.append(Config(config_name(name, k, n, t, pt), PRESETS[name], top_k=k,
                                              max_per_turn=pt, neighbors=n, neighbor_from_top=t))
    return configs


def ablation_configs(base: str = "V2", k: int = 10) -> list[Config]:
    """One axis at a time, from `base`@k (docs/rag-planv2.md 5)."""
    p = PRESETS[base]
    b = f"{base}@{k}"
    return [
        Config(f"{b} dedup=off", p, k, dedup="off"),
        Config(f"{b} dedup=hash", p, k, dedup="hash"),
        Config(f"{b} per-turn=2", p, k, max_per_turn=2),
        Config(f"{b} recent=0.4", p, k, recent_max_share=0.4),
        Config(f"{b} recent=0.8", p, k, recent_max_share=0.8),
        Config(f"{b} query=full", p, k, dense_query="full"),
        Config(f"{b} no-header", p, k, representation="plain"),
        Config(f"{b} idf=session", p, k, idf="session"),
        Config(f"V1@{k} no-header", PRESETS["V1"], k, representation="plain"),
    ]


# ------------------------------------------------------------------ evaluation
def dense_query_text(prompt: str, mode: str) -> str:
    if mode == "full":
        return tail_tokens(prompt.strip(), DENSE_QUERY_MAX_TOKENS)
    return build_dense_query(prompt)


async def query_vectors(indexed: list[IndexedSession], embedder: CachedEmbedder) -> dict[tuple, list[float]]:
    """(question id, session id, dense_query mode) -> the unit query vector."""
    queries = {(q.id, s.session.id, mode): dense_query_text(q.prompt, mode)
               for s in indexed for q in s.session.questions for mode in ("extracted", "full")}
    keys = list(queries)
    return dict(zip(keys, (_unit(v) for v in await embedder.embed([queries[k] for k in keys]))))


def corpus_stats(indexed: list[IndexedSession], representation: str = "header") -> TermStats:
    """BM25 document frequencies over every chunk of every session: the stand-in for the
    production corpus (all indexed chunks)."""
    chunks = [c for s in indexed for c in s.chunks]
    return TermStats(len(chunks), Counter(t for c in chunks for t in set(c.terms_by[representation])))


def retrieve(s: IndexedSession, q: Question, cfg: Config, settings: RetrievalSettings,
             query_vector: list[float], corpus: TermStats | None = None) -> list[int]:
    """The pool (chunk ids, best first) the router would get for `q` under `cfg`; [] for V0.
    `corpus`: BM25's IDF statistics, used when cfg.idf == "corpus"."""
    if not cfg.preset.retrieves:
        return []
    candidates = [Candidate(c.id, c.turn, c.ord, c.content_hash, c.terms_by[cfg.representation],
                            cosine_distance(query_vector, c.vector_by[cfg.representation])
                            if cfg.preset.dense else None, c.source)
                  for c in s.chunks]
    params = RankParams(candidates=settings.candidates, pool=settings.pool, dedup=cfg.dedup,
                        jaccard=settings.dedup_jaccard, neighbor_turns=cfg.neighbors,
                        neighbor_from_top=cfg.neighbor_from_top, gate_similarity=cfg.gate,
                        min_similarity=cfg.min_sim)
    return [r.id for r in rank(candidates, analyze(q.prompt) if cfg.preset.bm25 else [], cfg.preset, params,
                               corpus if cfg.idf == "corpus" else None)]


async def evaluate(indexed: list[IndexedSession], configs: list[Config], models: list[ModelProfile],
                   embedder: CachedEmbedder, settings: RetrievalSettings | None = None) -> list[dict]:
    settings = settings or RetrievalSettings()
    vectors = await query_vectors(indexed, embedder)
    corpus = {rep: corpus_stats(indexed, rep) for rep in {c.representation for c in configs}}

    rows = []
    for s in indexed:
        turn_texts = recent_turn_texts(s, settings)
        by_id = {c.id: c for c in s.chunks}
        for q in s.session.questions:
            for cfg in configs:
                rows.append(_evaluate_one(s, q, cfg, models, settings, turn_texts, by_id,
                                          vectors[(q.id, s.session.id, cfg.dense_query)],
                                          corpus[cfg.representation]))
    return rows


def _evaluate_one(s: IndexedSession, q: Question, cfg: Config, models: list[ModelProfile],
                  settings: RetrievalSettings, turn_texts: list[TurnText], by_id: dict[int, IndexedChunk],
                  query_vector: list[float], corpus: TermStats | None = None) -> dict:
    gold, outdated = s.gold_ids[q.id], s.outdated_ids[q.id]
    pool_ids = retrieve(s, q, cfg, settings, query_vector, corpus)
    top = pool_ids[:cfg.top_k]
    first_gold = next((i for i, cid in enumerate(pool_ids, start=1) if cid in gold), None)

    row = {
        **cfg.describe(), "session": s.session.id, "category": s.session.category,
        "session_turns": len(s.session.turns), "question": q.id, "type": q.type, "lang": q.lang,
        "gold_turns": sorted({g.turn for g in q.gold}), "gold_kinds": s.gold_kinds[q.id],
        "retrieval": None, "context": {},
    }
    if cfg.preset.retrieves:
        row["retrieval"] = {
            "hit": any(cid in gold for cid in top),
            "rr": 1.0 / first_gold if first_gold else 0.0,
            "distinct_turns": len({by_id[cid].turn for cid in top}),
            "outdated": any(cid in outdated for cid in top),
            "outdated_above_gold": bool(outdated) and _rank_of(pool_ids, outdated) < _rank_of(pool_ids, gold),
        }

    for m in models:
        _, row["context"][m.name] = assemble(s, q, cfg, m, settings, turn_texts, by_id, pool_ids)
    return row


def recent_turn_texts(s: IndexedSession, settings: RetrievalSettings) -> list[TurnText]:
    """What R1 returns for a question asked after the session's last turn: its last
    RAG_RECENT_TURNS_MAX turns, oldest first, numbered as in the session."""
    recent = s.session.turns[-settings.recent_turns_max:]
    first = len(s.session.turns) - len(recent) + 1
    return [TurnText(first + i, t.user, t.assistant) for i, t in enumerate(recent)]


def assemble(s: IndexedSession, q: Question, cfg: Config, m: ModelProfile, settings: RetrievalSettings,
             turn_texts: list[TurnText], by_id: dict[int, IndexedChunk],
             pool_ids: list[int]) -> tuple[AssembledContext, dict]:
    """The context the router would send to a model with profile `m` (build_messages, as in
    production), and what reached it: the gold as a whole turn / a fragment / not at all."""
    gold, outdated = s.gold_ids[q.id], s.outdated_ids[q.id]
    fragments = [Fragment(c.id, c.turn, c.ord, c.content, c.header, c.header_from_answer)
                 for c in (by_id[cid] for cid in pool_ids)]
    ctx = build_messages(
        SYSTEM_PROMPT, turn_texts, q.prompt,
        Budget(m.context_window, m.max_output_tokens, margin=settings.context_margin,
               recent_max_share=cfg.recent_max_share, last_turn_max_share=settings.last_turn_max_share),
        fragments=fragments, top_k=cfg.top_k, max_per_turn=cfg.max_per_turn)
    whole = {t.turn for t in turn_texts[len(turn_texts) - ctx.recent_turns:]} if ctx.recent_turns else set()
    via = "whole" if {g.turn for g in q.gold} & whole else "fragment" if gold & set(ctx.fragment_ids) else "none"
    return ctx, {
        "present": via != "none", "via": via, "tokens": ctx.input_tokens,
        "recent_turns": ctx.recent_turns, "fragments": len(ctx.fragment_ids),
        "fragment_turns": len({by_id[i].turn for i in ctx.fragment_ids}),
        "outdated_fragment": bool(outdated & set(ctx.fragment_ids)),
    }


def _rank_of(pool_ids: list[int], ids: set[int]) -> float:
    return next((i for i, cid in enumerate(pool_ids) if cid in ids), math.inf)
