"""The RAG versions (docs/rag-planv2.md 5) as named presets: which retrieval methods run. The same
code runs every version; the numeric knobs (top-k, pool, dedup, ...) are RetrievalSettings and
are the same for all of them unless an experiment changes one.

    V0  recent turns only (no retrieval)
    V1  dense                                  V3  dense on the contextualized chunks
    V2  dense + BM25, fused with RRF           V4  hybrid on the contextualized chunks
                                               V5  V4 + a reranker

V3-V5 need the LLM-generated chunk context, which is produced offline only (4.8), so the router
accepts V0-V2 (`online`); the harness runs them all.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Preset:
    name: str
    dense: bool = False
    bm25: bool = False
    contextual: bool = False      # embedding_ctx / terms_ctx instead of embedding / terms
    rerank: bool = False

    @property
    def retrieves(self) -> bool:
        return self.dense or self.bm25

    @property
    def online(self) -> bool:
        """Runnable in the router: no LLM context per chunk, no reranker."""
        return not self.contextual and not self.rerank


PRESETS: dict[str, Preset] = {p.name: p for p in (
    Preset("V0"),
    Preset("V1", dense=True),
    Preset("V2", dense=True, bm25=True),
    Preset("V3", dense=True, contextual=True),
    Preset("V4", dense=True, bm25=True, contextual=True),
    Preset("V5", dense=True, bm25=True, contextual=True, rerank=True),
)}


def get_preset(name: str) -> Preset:
    try:
        return PRESETS[name]
    except KeyError:
        raise ValueError(f"unknown RAG variant {name!r} (known: {', '.join(PRESETS)})") from None
