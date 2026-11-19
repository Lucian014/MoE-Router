"""
Relevance thresholds: which RAG_GATE_SIMILARITY / RAG_MIN_SIMILARITY keeps the noise out of the
context of requests that need nothing from the history, without losing the gold of those that do.

    python scripts/rag_eval/threshold_eval.py                         # V2@10+n1, the default grid
    python scripts/rag_eval/threshold_eval.py --base V2@10+n1 --gates 0.40 0.45 0.50 --floors 0.35 0.40

Positives: every question of the datasets (--data, default data/sessions, data/multihop,
data/production): is the gold still in the context the router would build (4k and 32k windows)?
Negatives: every prompt of data/standalone.yaml asked in EVERY session of those datasets: is the
context clean (no fragment at all), how many fragments / tokens of excerpts are sent anyway?
Before the grid, the distribution of the best dense similarity (positives vs negatives) shows where a
gate could sit. Same code as production (harness.retrieve -> rank, assembly.build_messages).
"""
import argparse
import asyncio
import datetime
import json
import statistics
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import DATA_DIR, check, load_dataset
from harness import (CachedEmbedder, ModelProfile, _unit, assemble, config_name, corpus_stats, cosine_distance,
                     dense_query_text, index_sessions, parse_config_name, query_vectors, recent_turn_texts, retrieve)
from shared.config import DEFAULT_EMBEDDING_MODEL, RetrievalSettings
from shared.memory.assembly import Budget, Fragment, build_messages
from shared.memory.embedding import OllamaEmbeddingClient
from shared.prompts import SYSTEM_PROMPT

DATASETS = [DATA_DIR, DATA_DIR.parent / "multihop", DATA_DIR.parent / "production"]
STANDALONE = DATA_DIR.parent / "standalone.yaml"
WINDOWS = [ModelProfile("4k", 4096, 512), ModelProfile("32k", 32768, 2048)]
DEFAULT_GRID = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def best_similarity(s, vector) -> float:
    return max(1.0 - cosine_distance(vector, c.vector_by["header"]) for c in s.chunks)


def negative_context(s, prompt, cfg, m, settings, turn_texts, by_id, pool) -> tuple[int, int]:
    """(fragments, excerpt tokens) the router would send for a request with no gold."""
    fragments = [Fragment(c.id, c.turn, c.ord, c.content, c.header, c.header_from_answer)
                 for c in (by_id[i] for i in pool)]
    budget = Budget(m.context_window, m.max_output_tokens, margin=settings.context_margin,
                    recent_max_share=cfg.recent_max_share, last_turn_max_share=settings.last_turn_max_share)
    ctx = build_messages(SYSTEM_PROMPT, turn_texts, prompt, budget, fragments=fragments, top_k=cfg.top_k,
                         max_per_turn=cfg.max_per_turn)
    plain = build_messages(SYSTEM_PROMPT, turn_texts, prompt, budget, top_k=0)
    return len(ctx.fragment_ids), ctx.input_tokens - plain.input_tokens


def pct(xs) -> str:
    return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "—"


async def run(args) -> Path:
    settings = RetrievalSettings()
    embedder = CachedEmbedder(OllamaEmbeddingClient(args.embedding_host, args.embedding_model, timeout=600))
    sets = []
    for d in args.data:
        sessions = [s for s in load_dataset(d) if not check(s) and s.questions]
        indexed = await index_sessions(sessions, embedder, ("header",))
        sets.append((d.name, indexed, await query_vectors(indexed, embedder), corpus_stats(indexed)))
    negatives = yaml.safe_load(args.standalone.read_text(encoding="utf-8"))["prompts"]
    neg_vecs = [_unit(v) for v in await embedder.embed([dense_query_text(n["prompt"], "extracted") for n in negatives])]

    base = parse_config_name(args.base)
    configs = [base]
    for g in args.gates:
        configs.append(parse_config_name(config_name(base.preset.name, base.top_k, base.neighbors,
                                                     base.neighbor_from_top, base.max_per_turn, gate=g)))
    for f in args.floors:
        configs.append(parse_config_name(config_name(base.preset.name, base.top_k, base.neighbors,
                                                     base.neighbor_from_top, base.max_per_turn, min_sim=f)))

    # where would a gate sit: the best similarity in the session, positives vs negatives
    pos_best, neg_best = {}, {"far": [], "near": []}
    for name, indexed, vecs, _ in sets:
        pos_best[name] = [best_similarity(s, vecs[(q.id, s.session.id, "extracted")])
                          for s in indexed for q in s.session.questions]
        for s in indexed:
            for n, v in zip(negatives, neg_vecs):
                neg_best[n["kind"]].append(best_similarity(s, v))
    lines = ["## Best dense similarity in the session (what the gate compares)", "",
             "| queries | n | p5 | p25 | median | p75 | p95 |", "|---|---|---|---|---|---|---|"]
    for label, xs in [*((f"questions: {k}", v) for k, v in pos_best.items()),
                      *((f"standalone: {k}", v) for k, v in neg_best.items())]:
        q = statistics.quantiles(xs, n=20)
        lines.append(f"| {label} | {len(xs)} | {q[0]:.3f} | {q[4]:.3f} | {statistics.median(xs):.3f} | "
                     f"{q[14]:.3f} | {q[18]:.3f} |")

    head = ["config"] + [f"gold {n} {m.name}" for n, *_ in sets for m in WINDOWS] + \
           ["clean far", "clean near", "fragments (standalone, 4k)", "excerpt tokens (standalone, 4k)",
            "fragments (questions, 4k)"]
    table = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    rows = []
    for cfg in configs:
        row = {"config": cfg.name}
        cells = [cfg.name]
        q_frags = []
        for name, indexed, vecs, corpus in sets:
            hits = {m.name: [] for m in WINDOWS}
            for s in indexed:
                turn_texts = recent_turn_texts(s, settings)
                by_id = {c.id: c for c in s.chunks}
                for q in s.session.questions:
                    pool = retrieve(s, q, cfg, settings, vecs[(q.id, s.session.id, cfg.dense_query)], corpus)
                    for m in WINDOWS:
                        _, info = assemble(s, q, cfg, m, settings, turn_texts, by_id, pool)
                        hits[m.name].append(info["present"])
                        if m.name == "4k":
                            q_frags.append(info["fragments"])
            for m in WINDOWS:
                row[f"gold {name} {m.name}"] = sum(hits[m.name]) / len(hits[m.name])
                cells.append(pct(hits[m.name]))
        clean = {"far": [], "near": []}
        frags, toks = [], []
        for name, indexed, _, corpus in sets:
            for s in indexed:
                turn_texts = recent_turn_texts(s, settings)
                by_id = {c.id: c for c in s.chunks}
                for n, v in zip(negatives, neg_vecs):
                    pseudo = type("Q", (), {"prompt": n["prompt"]})
                    pool = retrieve(s, pseudo, cfg, settings, v, corpus)
                    f, t = negative_context(s, n["prompt"], cfg, WINDOWS[0], settings, turn_texts, by_id, pool)
                    clean[n["kind"]].append(f == 0)
                    frags.append(f)
                    toks.append(t)
        row.update({"clean far": sum(clean["far"]) / len(clean["far"]),
                    "clean near": sum(clean["near"]) / len(clean["near"]),
                    "standalone fragments": statistics.mean(frags), "standalone tokens": statistics.mean(toks),
                    "question fragments": statistics.mean(q_frags)})
        cells += [pct(clean["far"]), pct(clean["near"]), f"{statistics.mean(frags):.1f}",
                  f"{statistics.mean(toks):.0f}", f"{statistics.mean(q_frags):.1f}"]
        table.append("| " + " | ".join(cells) + " |")
        rows.append(row)
        print("  " + " | ".join(cells), flush=True)

    n_neg = {k: len(v) for k, v in neg_best.items()}
    report = "\n".join([
        "# Relevance thresholds", "",
        f"Base {base.name}; gate = no fragments when the session's best chunk is below the similarity, "
        f"floor (+s) = each fragment at least that similar (next-turn neighbors exempt). Questions: "
        + ", ".join(f"{n} {sum(len(s.session.questions) for s in ix)}" for n, ix, *_ in sets)
        + f". Standalone requests: {len(negatives)} prompts x every session = {n_neg['far']} far + "
        f"{n_neg['near']} near pairs. *clean* = no fragment at all in the 4k context.", "",
        *lines, "", "## Grid", "", *table, ""])
    out = args.out or HERE / "results" / ("thresholds-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.md").write_text(report, encoding="utf-8")
    (out / "rows.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print("\n" + report + f"\n[OK] {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, nargs="+", default=DATASETS)
    ap.add_argument("--standalone", type=Path, default=STANDALONE)
    ap.add_argument("--base", default="V2@10+n1", help="the config the thresholds are added to")
    ap.add_argument("--gates", type=float, nargs="*", default=DEFAULT_GRID, help="RAG_GATE_SIMILARITY values")
    ap.add_argument("--floors", type=float, nargs="*", default=DEFAULT_GRID, help="RAG_MIN_SIMILARITY values")
    ap.add_argument("--embedding-host", default="http://localhost:11434")
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--out", type=Path)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
