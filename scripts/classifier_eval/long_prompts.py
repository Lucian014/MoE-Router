"""
How the router's classifiers route LONG prompts: a document (prose, code or a log) with the
request before or after it ("Rezumă textul de mai jos." + 40 pages).

    python scripts/classifier_eval/long_prompts.py                         # every case, every strategy
    python scripts/classifier_eval/long_prompts.py --strategies raw         # only what the router does today
    python scripts/classifier_eval/long_prompts.py --sizes 800 --variants 1 # a quick run
    python scripts/classifier_eval/long_prompts.py --catalog path/models.yaml   # + the model each case gets
    # ablations: windows x strategies x edges thresholds, one run, one table
    python scripts/classifier_eval/long_prompts.py --max-lens 512 1024 2048 --strategies raw edges \\
        --edge-max-tokens 60 100 --sizes 200 800 2000 4000

The classifiers see at most max_len = 512 tokens, of which up to head_max_len = 256 go to Laya's
question and options (checkpoints/*/rl_agent_config.json): the prompt gets the rest, ~250 tokens.
Laya keeps the START of a longer prompt and drops the rest without a warning, so a request written
after the document is never seen.

Cases = request x document x size x position:
    documents   built from the RAG dataset (scripts/rag_eval/data/sessions): its texts are segmented
                (shared/memory/segment.py) and pieces of one kind (prose, code) are joined up to a
                target length in classifier tokens; logs are generated (the dataset has almost none)
    position    start ("Rezumă textul de mai jos." + document) or end (document + "Rezumă textul de mai sus.")
    alone       the request without a document: what the classifier says when nothing is cut
Each request has the domain it should be routed to (domain_schema.LABELS).

Strategies = what the classifiers are given:
    raw         the prompt as it is; Laya keeps its first ~250 tokens (the router today)
    head_tail   the first and the last half of the budget, "[…]" between
    edges       a short first / last paragraph (where the request usually is) + the start of the rest
    windows     Laya's predict_long: overlapping windows over the whole prompt, the most confident decides

Writes results/<timestamp>/{cases.jsonl, report.md} next to this file and prints the report.
"""
import argparse
import datetime
import json
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from router.classifier_input import EDGE_MAX_TOKENS, edges, head_tail, paragraphs  # noqa: E402
from shared.memory.segment import segment  # noqa: E402

SESSIONS_DIR = ROOT / "scripts" / "rag_eval" / "data" / "sessions"
SIZES = (200, 800, 2000)               # document length, in classifier tokens; 200 still fits
STRATEGIES = ("raw", "head_tail", "edges", "windows")

Count = Callable[[str], int]


# ------------------------------------------------------------------ requests
@dataclass(frozen=True)
class Request:
    label: str           # the domain it should be routed to
    doc: str             # prose | code | log
    start: str           # written before the document
    end: str             # written after it


REQUESTS = (
    Request("summarization", "prose", "Rezumă textul de mai jos în 5 puncte.", "Rezumă textul de mai sus în 5 puncte."),
    Request("summarization", "prose", "Care sunt ideile principale din articolul următor? Fă un rezumat scurt.",
            "Care sunt ideile principale din articolul de mai sus? Fă un rezumat scurt."),
    Request("summarization", "prose", "Summarize the text below in three bullet points.",
            "Summarize the text above in three bullet points."),
    Request("translation", "prose", "Tradu în engleză textul de mai jos.", "Tradu în engleză textul de mai sus."),
    Request("translation", "prose", "Translate the following text into French.",
            "Translate the text above into French."),
    Request("text-rewrite", "prose", "Rescrie textul de mai jos ca pe un e-mail formal către conducere.",
            "Rescrie textul de mai sus ca pe un e-mail formal către conducere."),
    Request("brainstorming", "prose", "Dă-mi 5 idei de titluri pentru articolul de mai jos.",
            "Dă-mi 5 idei de titluri pentru articolul de mai sus."),
    Request("creative-writing", "prose", "Scrie o poezie scurtă inspirată din textul de mai jos.",
            "Scrie o poezie scurtă inspirată din textul de mai sus."),
    Request("coding", "code", "Găsește bug-ul din codul de mai jos.", "Găsește bug-ul din codul de mai sus."),
    Request("coding", "code", "Explică pe scurt ce face codul următor.", "Explică pe scurt ce face codul de mai sus."),
    Request("coding", "code", "Review this code and suggest improvements.",
            "Review the code above and suggest improvements."),
    Request("coding", "log", "De ce crapă aplicația? Iată logul:", "Ce înseamnă eroarea asta și cum o repar?"),
    Request("coding", "log", "Why does this pod keep restarting? Here are the logs:",
            "Why does this pod keep restarting?"),
)


# ------------------------------------------------------------------ documents
TECHNICAL_SESSIONS = ("coding", "math")     # their prose is about code: a "text" document must not be
MIN_PROSE_WORDS = 40                        # shorter paragraphs are mostly the dataset's own requests


def dataset_pieces() -> dict[str, list[str]]:
    """Code segments of every session; long prose paragraphs of the non-technical ones (without the
    dataset's requests: "Rezumă-mi articolul ăsta:" must not end up inside a document). File order."""
    pieces: dict[str, list[str]] = defaultdict(list)
    for path in sorted(SESSIONS_DIR.glob("*.yaml")):
        session = yaml.safe_load(path.read_text(encoding="utf-8"))
        technical = session.get("category") in TECHNICAL_SESSIONS
        for turn in session["turns"]:
            for text in (turn["user"], turn["assistant"]):
                for seg in segment(text):
                    if seg.kind == "code" and len(seg.text.split()) >= 15:
                        pieces["code"].append(seg.text.strip())
                    elif seg.kind == "prose" and not technical:
                        pieces["prose"] += [p for p in paragraphs(seg.text) if len(p.split()) >= MIN_PROSE_WORDS]
    return pieces


def log_lines(n: int, seed: int) -> list[str]:
    """A plausible application log: access lines and warnings, then a traceback (deterministic)."""
    paths = ("/items/42", "/login", "/orders?page=2", "/health", "/users/me", "/reports/monthly")
    lines = []
    for i in range(n):
        k = i + seed * 7
        second = k % 60
        minute = 3 + k // 60
        ts = f"2026-05-01T12:{minute:02d}:{second:02d}.{(k * 37) % 1000:03d}Z"
        if k % 11 == 5:
            lines.append(f"{ts} WARN  db.pool  connection slow: {(k * 13) % 900 + 100} ms (pool size 10, in use 9)")
        elif k % 17 == 3:
            lines.append(f"{ts} INFO  worker   job {k:05d} retried after timeout (attempt 2/3)")
        else:
            status = (200, 200, 201, 304, 404)[k % 5]
            lines.append(f'{ts} INFO  uvicorn.access 10.0.{k % 8}.{k % 200} - "GET {paths[k % len(paths)]} '
                         f'HTTP/1.1" {status}')
    lines += [
        "2026-05-01T12:59:58.112Z ERROR app      unhandled exception in request",
        "Traceback (most recent call last):",
        '  File "/app/main.py", line 88, in get_report',
        "    rows = await repo.monthly(user.id)",
        '  File "/app/repository.py", line 41, in monthly',
        "    return await self.session.execute(stmt)",
        "sqlalchemy.exc.OperationalError: (psycopg.OperationalError) connection to server at "
        '"moe-postgres" (10.43.0.12), port 5432 failed: FATAL: sorry, too many clients already',
    ]
    return lines


def build_document(kind: str, target: int, variant: int, pieces: dict[str, list[str]], count: Count) -> str:
    """About `target` classifier tokens of one kind; variant v starts further into the material."""
    if kind == "log":
        lines, n = [], 10
        while True:
            lines = log_lines(n, variant)
            if count("\n".join(lines)) >= target or n > 5000:
                return "\n".join(lines)
            n = int(n * 1.5)
    material = pieces[kind]
    if not material:
        raise SystemExit(f"no {kind} material in {SESSIONS_DIR}")
    start = (variant * len(material)) // 3
    out: list[str] = []
    for i in range(len(material)):
        out.append(material[(start + i) % len(material)])
        if count("\n\n".join(out)) >= target:
            break
    return "\n\n".join(out)


@dataclass(frozen=True)
class Case:
    id: str
    label: str
    doc: str
    size: int | None     # None: the request alone
    position: str        # start | end | alone
    prompt: str
    tokens: int


def build_cases(pieces: dict[str, list[str]], count: Count, sizes=SIZES, variants: int = 2,
                requests=REQUESTS) -> list[Case]:
    cases: list[Case] = []
    docs: dict[tuple[str, int, int], str] = {}
    for r_index, req in enumerate(requests):
        cases.append(Case(f"r{r_index}-alone", req.label, req.doc, None, "alone", req.start, count(req.start)))
        for size in sizes:
            for v in range(variants):
                key = (req.doc, size, v)
                if key not in docs:
                    docs[key] = build_document(req.doc, size, v, pieces, count)
                doc = docs[key]
                for position, prompt in (("start", f"{req.start}\n\n{doc}"), ("end", f"{doc}\n\n{req.end}")):
                    cases.append(Case(f"r{r_index}-{req.doc}{size}-v{v}-{position}", req.label, req.doc, size,
                                      position, prompt, count(prompt)))
    return cases


# ------------------------------------------------------------------ strategies (what the classifier is given)
# the router's own rules (router/classifier_input.py): what is measured here is what runs there
SHAPERS = {"head_tail": head_tail, "edges": edges}


# ------------------------------------------------------------------ running the classifiers
@dataclass(frozen=True)
class Config:
    """One ablation point: what the classifiers are given and how wide their window is."""
    strategy: str
    max_len: int
    edge_max: int = EDGE_MAX_TOKENS          # edges only

    @property
    def name(self) -> str:
        edge = f"/{self.edge_max}" if self.strategy == "edges" and self.edge_max != EDGE_MAX_TOKENS else ""
        return f"{self.strategy}{edge}@{self.max_len}"


def configs_from(strategies, max_lens, edge_maxes) -> list[Config]:
    out = []
    for max_len in max_lens:
        for strategy in strategies:
            for edge_max in (edge_maxes if strategy == "edges" else [EDGE_MAX_TOKENS]):
                out.append(Config(strategy, max_len, edge_max))
    return out


def budget_of(max_len: int, head_max_len: int) -> int:
    return max(64, max_len - head_max_len - 8)           # predict_long's per-question state budget


def given_text(case: "Case", config: Config, budget: int, count: Count) -> str:
    if config.strategy == "raw":
        return case.prompt
    if config.strategy == "edges":
        return edges(case.prompt, budget, count, config.edge_max)
    return SHAPERS[config.strategy](case.prompt, budget, count)


def run(cases: list[Case], configs: list[Config], domain_agent, complexity_agent, domain_schema, complexity_schema,
        head_max_len: int, count: Count, threshold: float, catalog=None, select_model=None) -> list[dict]:
    """Every case under every config. A (text, window) pair is classified once: at a wide window a
    short document gives the same text under every strategy."""
    rows = []
    memo: dict[tuple, tuple] = {}
    started = time.perf_counter()
    total = len(cases) * len(configs)
    done = 0
    for config in configs:
        budget = budget_of(config.max_len, head_max_len)
        for case in cases:
            done += 1
            if case.position == "alone" and config.strategy != "raw":
                continue                       # nothing to cut: every strategy is the same
            if config.strategy == "windows":
                key = ("windows", case.prompt, budget)
                given = None
            else:
                given = given_text(case, config, budget, count)
                key = ("text", given, config.max_len)
            if key not in memo:
                t = time.perf_counter()
                if config.strategy == "windows":
                    d = domain_agent.predict_long(case.prompt, domain_schema.QUESTIONS, window=budget)
                    c = complexity_agent.predict_long(case.prompt, complexity_schema.QUESTIONS, window=budget)
                else:
                    d = domain_agent.predict(given, domain_schema.QUESTIONS, max_len=config.max_len)
                    c = complexity_agent.predict(given, complexity_schema.QUESTIONS, max_len=config.max_len)
                memo[key] = (d, c, round((time.perf_counter() - t) * 1000, 1))
            d, c, ms = memo[key]
            probs = {k: float(v) for k, v in d["answers"][domain_schema.QID]["probabilities"].items()}
            predicted = max(probs, key=probs.get)
            score = float(c["answers"][complexity_schema.QID]["score"])
            level = "high" if score >= threshold else "low"
            row = {
                "case": case.id, "config": config.name, "strategy": config.strategy, "max_len": config.max_len,
                "edge_max": config.edge_max, "label": case.label, "doc": case.doc, "size": case.size,
                "position": case.position, "prompt_tokens": case.tokens, "given_tokens": count(given) if given else None,
                "predicted": predicted, "confidence": round(probs[predicted], 4), "correct": predicted == case.label,
                "probabilities": {k: round(v, 4) for k, v in probs.items()},
                "complexity_score": round(score, 4), "complexity": level, "ms": ms,
            }
            if config.strategy == "windows":
                row["windows"] = (d.get("usage") or {}).get("windows")
            if given is not None and given != case.prompt:
                row["given"] = given
            if catalog is not None:
                row["model"] = select_model(catalog, predicted, level)[0].model_id
                row["expected_model"] = select_model(catalog, case.label, level)[0].model_id
            rows.append(row)
            if done % 25 == 0 or done == total:
                print(f"[*] {done}/{total} ({config.name}, {time.perf_counter() - started:.0f} s, "
                      f"{len(memo)} classified)", flush=True)
    return rows


# ------------------------------------------------------------------ report
def _pct(rows) -> str:
    return f"{100 * sum(r['correct'] for r in rows) / len(rows):.0f}% ({sum(r['correct'] for r in rows)}/{len(rows)})" \
        if rows else "—"


def _share(rows) -> str:
    return f"{100 * sum(r['correct'] for r in rows) / len(rows):.0f}%" if rows else "—"


def _high(rows) -> str:
    return f"{100 * sum(r['complexity'] == 'high' for r in rows) / len(rows):.0f}%" if rows else "—"


def _median_ms(rows) -> str:
    ms = sorted(r["ms"] for r in rows)
    return f"{ms[len(ms) // 2]:.0f}" if ms else "—"


def report(rows: list[dict], meta: dict) -> str:
    configs = list(dict.fromkeys(r["config"] for r in rows if r["position"] != "alone"))
    sizes = sorted({r["size"] for r in rows if r["size"] is not None})
    long_rows = [r for r in rows if r["position"] != "alone"]
    by = {c: [r for r in long_rows if r["config"] == c] for c in configs}
    out = ["# Classifiers on long prompts", "",
           f"Windows (max_len) {meta['max_lens']}, head_max_len {meta['head_max_len']}: the prompt gets "
           f"max_len - head_max_len - 8 tokens. {meta['cases']} cases, {meta['requests']} requests, "
           f"document sizes {sizes} (classifier tokens), {meta['variants']} document variant(s). "
           f"Config = strategy[/edge threshold]@max_len.", ""]

    out += ["## Overview", "",
            "| config | all | request at start | request at end | " + " | ".join(f"{s} tok" for s in sizes)
            + " | complexity high | median ms |",
            "|---|---|---|---|" + "---|" * len(sizes) + "---|---|"]
    for c in configs:
        rs = by[c]
        cells = [_share(rs), _share([r for r in rs if r["position"] == "start"]),
                 _share([r for r in rs if r["position"] == "end"])]
        cells += [_share([r for r in rs if r["size"] == size]) for size in sizes]
        cells += [_high(rs), _median_ms(rs)]
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    out.append("")

    alone = [r for r in rows if r["position"] == "alone"]
    out += ["## The request alone (nothing cut)", ""]
    for max_len in meta["max_lens"]:
        rs = [r for r in alone if r["max_len"] == max_len]
        wrong = ", ".join(f"{r['label']} → {r['predicted']}" for r in rs if not r["correct"]) or "none wrong"
        out.append(f"- max_len {max_len}: {_pct(rs)}; complexity high {_high(rs)}; {wrong}")
    out.append("")

    out += ["## By document kind and position", "",
            "| config | " + " | ".join(f"{d} / {p}" for d in ("prose", "code", "log") for p in ("start", "end")) + " |",
            "|---|" + "---|" * 6]
    for c in configs:
        out.append(f"| {c} | " + " | ".join(_share([r for r in by[c] if r["doc"] == d and r["position"] == p])
                                           for d in ("prose", "code", "log") for p in ("start", "end")) + " |")
    out.append("")

    out += ["## Per expected domain", "", "| config | " + " | ".join(meta["labels_used"]) + " |",
            "|---|" + "---|" * len(meta["labels_used"])]
    for c in configs:
        out.append(f"| {c} | " + " | ".join(_share([r for r in by[c] if r["label"] == lab])
                                           for lab in meta["labels_used"]) + " |")
    out.append("")

    out += ["## What the wrong ones became", ""]
    for c in configs:
        confusion = Counter((r["label"], r["predicted"]) for r in by[c] if not r["correct"])
        top = ", ".join(f"{a} → {b} ×{n}" for (a, b), n in confusion.most_common(6)) or "none"
        out.append(f"- **{c}**: {top}")
    out.append("")

    if any("model" in r for r in rows):
        out += ["## Routing (with --catalog)", "", "| config | same model as the expected domain |", "|---|---|"]
        for c in configs:
            same = sum(r["model"] == r["expected_model"] for r in by[c])
            out.append(f"| {c} | {100 * same / len(by[c]):.0f}% ({same}/{len(by[c])}) |")
        out.append("")
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strategies", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    ap.add_argument("--max-lens", nargs="+", type=int, default=[512],
                    help="the classifiers' windows to try, in tokens (the checkpoints were trained at 512; "
                         "the encoder, mmBERT, takes up to 8192)")
    ap.add_argument("--edge-max-tokens", nargs="+", type=int, default=[EDGE_MAX_TOKENS],
                    help="edges: how short a first / last paragraph must be to count as the request")
    ap.add_argument("--sizes", nargs="+", type=int, default=list(SIZES), help="document lengths, classifier tokens")
    ap.add_argument("--variants", type=int, default=2, help="documents per kind and size")
    ap.add_argument("--catalog", type=Path, help="models YAML ({models: [...]}): also report the routed model")
    ap.add_argument("--out", type=Path, default=HERE / "results")
    args = ap.parse_args()

    from router import classifier, complexity_schema, domain_schema
    from shared.config import get_classifier_settings

    classifier.load_classifiers()
    # the router's own Laya agents (same checkpoints, same device); predict_long needs the agent itself
    domain_agent, complexity_agent = classifier._DOMAIN.agent, classifier._COMPLEXITY.agent
    tok = domain_agent.tok

    def count(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    head_max_len = domain_agent.cfg.get("head_max_len", 192)
    catalog = select_model = None
    if args.catalog:
        from router.routing_rule import load_catalog, select_model
        catalog = load_catalog(args.catalog)

    configs = configs_from(args.strategies, args.max_lens, args.edge_max_tokens)
    cases = build_cases(dataset_pieces(), count, sizes=tuple(args.sizes), variants=args.variants)
    print(f"[*] {len(cases)} cases x {len(configs)} configs: {', '.join(c.name for c in configs)}")
    rows = run(cases, configs, domain_agent, complexity_agent, domain_schema, complexity_schema,
               head_max_len, count, get_classifier_settings().complexity_threshold, catalog, select_model)

    meta = {"max_lens": args.max_lens, "head_max_len": head_max_len, "cases": len(cases),
            "requests": len(REQUESTS), "variants": args.variants,
            "labels_used": [lab for lab in domain_schema.LABELS if any(r.label == lab for r in REQUESTS)]}
    text = report(rows, meta)
    out = args.out / datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "cases.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    (out / "run.json").write_text(json.dumps({**meta, "configs": [c.name for c in configs],
                                              "sizes": args.sizes}, indent=2), encoding="utf-8")
    (out / "report.md").write_text(text, encoding="utf-8")
    print("\n" + text)
    print(f"[OK] {out}")


if __name__ == "__main__":
    main()
