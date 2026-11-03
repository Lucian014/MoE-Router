"""
The tables of an evaluation run (docs/rag-planv2.md 6), as Markdown.

    python scripts/rag_eval/report.py results/<run>        # rewrite report.md from results.jsonl

Retrieval (independent of the model): Recall@k (a gold chunk in the top k), MRR (over the pool),
distinct turns in the top k, an outdated version in the top k. In context (per model profile):
the gold reached the model, as a whole turn or as a fragment; context tokens; fragments.
V3-V5 need the offline chunk context (step 9) and stay empty until then.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

PRESET_ROWS = ["V0", "V1", "V2", "V3", "V4", "V5"]


def _mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else None


def _pct(x):
    return "—" if x is None else f"{100 * x:.0f}%"


def _num(x, digits=2):
    return "—" if x is None else f"{x:.{digits}f}"


def _table(header: list[str], rows: list[list[str]]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines)


def _group(rows, key):
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)
    return groups


def _retrieval_cells(rs):
    ret = [r["retrieval"] for r in rs if r["retrieval"]]
    if not ret:
        return ["—", "—", "—", "—"]
    return [_pct(_mean(x["hit"] for x in ret)), _num(_mean(x["rr"] for x in ret)),
            _num(_mean(x["distinct_turns"] for x in ret), 1), _pct(_mean(x["outdated"] for x in ret))]


def _context_cells(rs, model):
    ctx = [r["context"][model] for r in rs if model in r["context"]]
    return [_pct(_mean(c["present"] for c in ctx)), _num(_mean(c["tokens"] for c in ctx), 0),
            _num(_mean(c["fragments"] for c in ctx), 1)]


def write_report(rows: list[dict], meta: dict, path: Path | None = None) -> str:
    models = [m["name"] for m in meta["models"]]
    configs = list(dict.fromkeys(r["config"] for r in rows))
    by_config = _group(rows, lambda r: r["config"])
    out = [
        "# RAG evaluation",
        "",
        f"{meta['sessions']} sessions, {meta['turns']} turns, {meta['questions']} questions, "
        f"{meta['chunks']} chunks (chunker {meta['chunker_version']}, {meta['embedding_model']}). "
        "Models: " + ", ".join(f"{m['name']} = {m['context_window']}/{m['max_output_tokens']}"
                               for m in meta["models"]) + ".",
        "",
        "## Main table",
        "",
    ]
    header = ["config", "Recall@k", "MRR", "turns in top-k", "outdated in top-k"]
    for m in models:
        header += [f"in context ({m})", f"tokens ({m})", f"fragments ({m})"]
    table_rows = []
    main = [c for c in configs if " " not in c]
    for preset in PRESET_ROWS:
        names = [c for c in main if c.split("@")[0] == preset]
        if not names:
            table_rows.append([preset] + ["—"] * (len(header) - 1))
        for c in names:
            rs = by_config[c]
            cells = [c] + _retrieval_cells(rs)
            for m in models:
                cells += _context_cells(rs, m)
            table_rows.append(cells)
    out += [_table(header, table_rows), ""]

    ablations = [c for c in configs if " " in c]
    if ablations:
        out += ["## Ablations (one axis at a time)", ""]
        rows_ = []
        for c in ablations:
            rs = by_config[c]
            cells = [c] + _retrieval_cells(rs)
            for m in models:
                cells += _context_cells(rs, m)
            rows_.append(cells)
        out += [_table(header, rows_), ""]

    focus = [c for c in main if c.endswith("@10")] or main
    for title, key in (("category", lambda r: r["category"]), ("question type", lambda r: r["type"]),
                       ("gold chunk kind", lambda r: ",".join(r["gold_kinds"]) or "?"),
                       ("language", lambda r: r["lang"]),
                       ("session length", lambda r: "≤ 20 turns" if r["session_turns"] <= 20 else "> 20 turns")):
        out += [f"## Per {title}", ""]
        h = [title, "questions"]
        for c in focus:
            h += [f"Recall ({c})"]
        for m in models:
            h += [f"in context V0 ({m})"] + [f"in context {c} ({m})" for c in focus]
        rows_ = []
        groups = _group([r for r in rows if r["config"] in focus + ["V0"]], key)
        for g in sorted(groups):
            rs = groups[g]
            cfg_rows = _group(rs, lambda r: r["config"])
            n = len(cfg_rows.get(focus[0], cfg_rows.get("V0", [])))
            cells = [g, str(n)]
            for c in focus:
                cells.append(_retrieval_cells(cfg_rows.get(c, []))[0])
            for m in models:
                cells.append(_context_cells(cfg_rows.get("V0", []), m)[0])
                cells += [_context_cells(cfg_rows.get(c, []), m)[0] for c in focus]
            rows_.append(cells)
        out += [_table(h, rows_), ""]

    misses = [r for r in rows if r["config"] in focus and r["retrieval"] and not r["retrieval"]["hit"]]
    if misses:
        out += [f"## Misses ({', '.join(focus)})", ""]
        out += [f"- {r['config']} {r['session']}/{r['question']} ({r['type']}, gold turns {r['gold_turns']})"
                for r in misses]
        out.append("")

    text = "\n".join(out)
    if path:
        path.write_text(text, encoding="utf-8")
    return text


def main():
    run = Path(sys.argv[1])
    rows = [json.loads(line) for line in (run / "results.jsonl").read_text(encoding="utf-8").splitlines()]
    meta = json.loads((run / "run.json").read_text(encoding="utf-8"))
    print(write_report(rows, meta, run / "report.md"))


if __name__ == "__main__":
    main()
