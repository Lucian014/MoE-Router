"""
Run the offline evaluation (docs/rag-planv2.md 6) and write the tables.

    # the embedding model on any Ollama, e.g. the cluster's:
    #   kubectl -n moe-router port-forward svc/moe-ollama 11434:11434
    python scripts/rag_eval/run.py                          # V0, V1, V2 x top-k 5/10/20
    python scripts/rag_eval/run.py --ablations              # + one axis at a time from V2@10
    python scripts/rag_eval/run.py --sessions 's0[1-3]*'    # a subset
    python scripts/rag_eval/run.py --models small=4096/512 mid=8192/1024 large=32768/2048

Writes results/<timestamp>/{results.jsonl, run.json, report.md}. Embeddings are cached in
.cache/, so a second run (other configs, other models) only embeds what is new.
"""
import argparse
import asyncio
import datetime
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import DATA_DIR, check, load_dataset
from harness import CachedEmbedder, ModelProfile, ablation_configs, evaluate, index_sessions, main_configs
from report import write_report
from shared.config import DEFAULT_EMBEDDING_MODEL
from shared.memory.chunking import index_version
from shared.memory.embedding import OllamaEmbeddingClient


def parse_model(spec: str) -> ModelProfile:
    """name=context_window/max_output_tokens"""
    name, limits = spec.split("=")
    window, output = limits.split("/")
    return ModelProfile(name, int(window), int(output))


async def run(args) -> Path:
    sessions = load_dataset(args.data, args.sessions)
    if not sessions:
        raise SystemExit(f"no sessions match {args.sessions!r} in {args.data}")
    problems = [p for s in sessions for p in check(s)]
    if problems:
        raise SystemExit("dataset problems (python scripts/rag_eval/dataset.py):\n  " + "\n  ".join(problems))

    embedder = CachedEmbedder(OllamaEmbeddingClient(args.embedding_host, args.embedding_model, timeout=600))
    models = [parse_model(m) for m in args.models]
    configs = (main_configs(args.top_k, args.neighbors, args.neighbor_from_top, args.per_turn)
               + (ablation_configs() if args.ablations else []))
    representations = ("header", "plain") if args.ablations else ("header",)

    t0 = time.perf_counter()
    indexed = await index_sessions(sessions, embedder, representations)
    index_s = time.perf_counter() - t0
    rows = await evaluate(indexed, configs, models, embedder)

    out = args.out or HERE / "results" / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "results.jsonl").open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {
        "data": str(args.data), "sessions": len(sessions), "turns": sum(len(s.turns) for s in sessions),
        "questions": sum(len(s.questions) for s in sessions), "chunks": sum(len(s.chunks) for s in indexed),
        "chunker_version": index_version(), "embedding_model": args.embedding_model,
        "models": [m.__dict__ for m in models], "configs": [c.describe() for c in configs],
        "index_seconds": round(index_s, 1),
    }
    (out / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    report = write_report(rows, meta, out / "report.md")
    print(report)
    print(f"\n[OK] {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embedding-host", default="http://localhost:11434")
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--data", type=Path, default=DATA_DIR, help="dataset folder (data/sessions/; data/multihop/ for the multi-hop set)")
    ap.add_argument("--sessions", default="*.yaml", help="glob in --data")
    ap.add_argument("--top-k", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--models", nargs="+", default=["small=4096/512", "large=32768/2048"],
                    help="name=context_window/max_output_tokens")
    ap.add_argument("--neighbors", type=int, nargs="+", default=[0],
                    help="RAG_NEIGHBOR_TURNS values to try (0 = off): each top chunk brings its next N turns' closest chunk")
    ap.add_argument("--neighbor-from-top", type=int, nargs="+", default=[3], help="RAG_NEIGHBOR_FROM_TOP values")
    ap.add_argument("--per-turn", type=int, nargs="+", default=[0], help="RAG_MAX_CHUNKS_PER_TURN values (0 = no cap)")
    ap.add_argument("--ablations", action="store_true")
    ap.add_argument("--out", type=Path)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
