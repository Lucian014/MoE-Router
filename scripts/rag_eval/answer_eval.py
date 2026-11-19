"""
End to end on the answers: do the models USE what retrieval puts in their context?

run.py / judge_retrieval.py stop at the context (did the gold reach the model?). This script goes one
step further, the way production does it: for every question x RAG config x model, the context is
assembled with the production code (harness.assemble -> shared/memory/assembly.build_messages), the
model answers it through the production engines (src/engine: Ollama /api/chat with num_ctx, vLLM
/v1/chat/completions), and an LLM judge scores the answer against the gold.

Two phases, so each runs where its resources are:

    # 1. answer -- where the models are (the A100): Ollama + vLLM + the embedding model
    python scripts/rag_eval/answer_eval.py answer \\
        --ollama http://127.0.0.1:11500 --vllm http://127.0.0.1:8100 \\
        --embedding-host http://127.0.0.1:11500 \\
        --models expert ollama:gemma3:1b vllm:gen
    python scripts/rag_eval/answer_eval.py answer ... --limit 3      # a smoke test: 3 questions

    # 2. judge -- where the judge is (GLM, scripts/rag_eval/.env), on the folder phase 1 wrote
    python scripts/rag_eval/answer_eval.py judge scripts/rag_eval/results/answers-<timestamp>

    # the tables again, without calling anything
    python scripts/rag_eval/answer_eval.py report scripts/rag_eval/results/answers-<timestamp>

Models (--models), each a "set" in the report:
    expert              every question goes to the catalog model the router's rule picks for the
                        question's category (routing_rule.select_model, --tier), with that model's
                        context_window / max_output_tokens: production without the classifiers
    <engine>:<model>    one model for every question (ollama:gemma3:1b, vllm:gen), with
                        --context-window / --max-output-tokens (default: the catalog default, 4096 / 512)

Configs (--configs): V0 (recent turns only), V1@k (dense), V2@k (dense + BM25).

The judge scores 1-5 against the gold excerpts (and, for a changed decision, the old version it must
not give). Report: mean score, share correct (score >= 4), split by whether the gold reached the
context (the model's part vs retrieval's part), old versions given, paired wins/losses between configs.

Answers are cached per model in .cache/answers-<model>.jsonl (key: the exact messages + limits), the
judge's scores in .cache/answer-judge-<model>.jsonl: an interrupted run resumes where it stopped.
Production sends no temperature, so neither does this: the models' own defaults.
"""
import argparse
import asyncio
import datetime
import hashlib
import json
import math
import re
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import DATA_DIR, Gold, Session, check, gold_ords, load_dataset
from harness import (CACHE_DIR, CachedEmbedder, Config, IndexedSession, ModelProfile, assemble,
                     corpus_stats, index_sessions, parse_config_name, query_vectors, recent_turn_texts, retrieve)
from judge_retrieval import (GLM_API_KEY, GLM_BASE_URL, GLM_INSECURE, GLM_MODEL, ENV_FILE,
                             openai_compatible)
from report import _mean, _num, _pct, _table

from engine import get_engine
from router.routing_rule import load_catalog, select_model
from shared.config import DEFAULT_CONTEXT_WINDOW, DEFAULT_EMBEDDING_MODEL, DEFAULT_MAX_OUTPUT_TOKENS, RetrievalSettings
from shared.memory.chunking import chunk_turn
from shared.memory.embedding import OllamaEmbeddingClient

ROOT = HERE.parents[1]
CATALOG = ROOT / "src" / "catalog" / "models.yaml"
DEFAULT_CONFIGS = ["V0", "V1@5", "V1@10", "V2@10"]
CORRECT = 4                  # a score >= this counts as a correct answer
PROMPT_VERSION = 1           # bump when JUDGE_PROMPT changes: the cached scores no longer apply
EXCERPT_CHARS = 2500         # per gold excerpt shown to the judge
ANSWER_CHARS = 6000          # of the answer shown to the judge

JUDGE_PROMPT = """You grade a chat assistant's answer. Late in a long conversation the user asked a \
question about something said earlier in it; the assistant answered from its memory of the conversation.

What the correct answer must contain -- the earlier turns where it was said:
{gold}
{outdated}
Question:
{question}

Answer to grade:
{answer}

Score 1-5:
5 = gives the information from those turns correctly and completely
4 = correct, with a minor detail missing or imprecise
3 = partly correct, or correct but too vague to act on
2 = mostly wrong, or gives the older version the conversation replaced
1 = wrong, invented, missing, or the assistant says it doesn't know / asks for it again

Reply with JSON only:
{{"score": <1-5>, "outdated": <true if the answer relies on the replaced version, else false>, "reason": "<one sentence>"}}"""

OUTDATED_BLOCK = """
The conversation CHANGED this later. This older version is NOT the correct answer any more:
{excerpts}
"""


# ------------------------------------------------------------------ what answers
@dataclass(frozen=True)
class ModelSpec:
    label: str                       # the report's "set": expert | <engine>:<model>
    engine: str | None = None        # None = expert (resolved per question)
    model: str | None = None


def parse_model_spec(spec: str) -> ModelSpec:
    if spec == "expert":
        return ModelSpec("expert")
    engine, sep, model = spec.partition(":")
    if not sep or not model:
        raise ValueError(f"model {spec!r}: expected 'expert' or <engine>:<model>, e.g. ollama:gemma3:1b")
    return ModelSpec(spec, engine, model)


def parse_config(name: str) -> Config:
    """V0, V1@10, V2@10+n1 (next-turn neighbors), V2@10+pt2 (chunks per turn): harness.config_name."""
    cfg = parse_config_name(name)
    if not cfg.preset.online:
        raise ValueError(f"config {name!r}: {cfg.preset.name} needs offline chunk context, production runs V0-V2")
    return cfg


@dataclass(frozen=True)
class Target:
    """Who answers one question: the engine, the model and its token budget."""
    label: str
    engine: str
    model: str
    profile: ModelProfile


def resolve(spec: ModelSpec, category: str, catalog, tier: str, window: int, output: int) -> Target:
    if spec.engine is None:
        m, _ = select_model(catalog, category, tier)
        return Target(spec.label, m.engine, m.model_id, ModelProfile(m.model_id, m.context_window, m.max_output_tokens))
    return Target(spec.label, spec.engine, spec.model, ModelProfile(spec.model, window, output))


# ------------------------------------------------------------------ caches
def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name)


class JsonlCache:
    """{key: value} appended to a .jsonl file, read back on start."""

    def __init__(self, path: Path | None):
        self.path, self.items = path, {}
        if path and path.exists():
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.items[row["k"]] = row["v"]

    @staticmethod
    def key(*parts) -> str:
        return hashlib.sha256(json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()).hexdigest()

    def get(self, key: str):
        return self.items.get(key)

    def put(self, key: str, value) -> None:
        self.items[key] = value
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"k": key, "v": value}, ensure_ascii=False) + "\n")


# ------------------------------------------------------------------ phase 1: answer
@dataclass
class Task:
    target: Target
    config: str
    session: IndexedSession
    question_id: str
    question_type: str
    messages: list[dict]
    context: dict


def build_tasks(indexed: list[IndexedSession], configs: list[Config], specs: list[ModelSpec],
                vectors: dict, settings: RetrievalSettings, catalog, tier: str, window: int, output: int,
                limit: int | None = None) -> list[Task]:
    corpus = corpus_stats(indexed)
    tasks, seen = [], 0
    for s in indexed:
        turn_texts = recent_turn_texts(s, settings)
        by_id = {c.id: c for c in s.chunks}
        for q in s.session.questions:
            if limit is not None and seen >= limit:
                return tasks
            seen += 1
            for cfg in configs:
                pool = retrieve(s, q, cfg, settings, vectors[(q.id, s.session.id, cfg.dense_query)], corpus)
                for spec in specs:
                    target = resolve(spec, s.session.category, catalog, tier, window, output)
                    ctx, info = assemble(s, q, cfg, target.profile, settings, turn_texts, by_id, pool)
                    tasks.append(Task(target, cfg.name, s, q.id, q.type, ctx.messages, info))
    return tasks


async def answer_all(tasks: list[Task], engines: dict, concurrency: int = 4,
                     cache_dir: Path | None = CACHE_DIR) -> list[dict]:
    """One model at a time (Ollama keeps it loaded), `concurrency` requests in flight."""
    by_model: dict[tuple[str, str], list[Task]] = defaultdict(list)
    for t in tasks:
        by_model[(t.target.engine, t.target.model)].append(t)

    rows = []
    for (engine_name, model), group in by_model.items():
        engine = engines[engine_name]
        cache = JsonlCache(cache_dir / f"answers-{_safe(engine_name)}-{_safe(model)}.jsonl" if cache_dir else None)
        sem = asyncio.Semaphore(concurrency)
        done = 0
        print(f"[*] {engine_name}:{model}: {len(group)} answers")

        async def one(t: Task) -> dict:
            nonlocal done
            p = t.target.profile
            key = JsonlCache.key(model, t.messages, p.max_output_tokens, p.context_window)
            result = cache.get(key)
            if result is None:
                async with sem:
                    start = time.perf_counter()
                    try:
                        r = await engine.chat(model=model, messages=t.messages,
                                              max_tokens=p.max_output_tokens, num_ctx=p.context_window)
                        result = {"answer": r.text, "truncated": r.truncated, "prompt_tokens": r.prompt_tokens,
                                  "completion_tokens": r.completion_tokens,
                                  "seconds": round(time.perf_counter() - start, 2)}
                        cache.put(key, result)
                    except Exception as e:          # recorded, not cached: a rerun asks again
                        result = {"error": f"{type(e).__name__}: {e}"}
            done += 1
            if done % 50 == 0 or done == len(group):
                print(f"    {done}/{len(group)}")
            return {"set": t.target.label, "engine": engine_name, "model": model, "session": t.session.session.id,
                    "category": t.session.session.category, "question": t.question_id, "type": t.question_type,
                    "config": t.config, "context_window": p.context_window, "max_output_tokens": p.max_output_tokens,
                    "context": t.context, **result}

        rows += await asyncio.gather(*(one(t) for t in group))
    return rows


def ready_sessions(pattern: str, data: Path = DATA_DIR) -> list[Session]:
    sessions = load_dataset(data, pattern)
    if not sessions:
        raise SystemExit(f"no sessions match {pattern!r} in {data}")
    ready = []
    for s in sessions:
        problems = check(s) + ([] if s.questions else [f"{s.id}: no questions yet"])
        if problems:
            print(f"[!] skipping {s.id}: " + "; ".join(problems[:3]))
        else:
            ready.append(s)
    if not ready:
        raise SystemExit("no session is ready (python scripts/rag_eval/dataset.py lists the problems)")
    return ready


async def run_answer(args) -> Path:
    specs = [parse_model_spec(m) for m in args.models]
    configs = [parse_config(c) for c in args.configs]
    catalog = load_catalog(args.catalog)
    sessions = ready_sessions(args.sessions, args.data)

    hosts = {"ollama": args.ollama, "vllm": args.vllm}
    settings = RetrievalSettings()
    embedder = CachedEmbedder(OllamaEmbeddingClient(args.embedding_host, args.embedding_model, timeout=600))
    indexed = await index_sessions(sessions, embedder)
    vectors = await query_vectors(indexed, embedder)
    tasks = build_tasks(indexed, configs, specs, vectors, settings, catalog, args.tier,
                        args.context_window, args.max_output_tokens, args.limit)

    needed = sorted({(t.target.engine, t.target.model) for t in tasks})
    engines = {}
    for name in sorted({e for e, _ in needed}):
        if not hosts.get(name):
            raise SystemExit(f"models on engine {name!r} need --{name} <url>")
        engines[name] = get_engine(name, hosts[name])
    missing = []
    for name, model in needed:                      # fail before hours of answers, not in the middle
        try:
            await engines[name].health(model)
        except Exception as e:
            missing.append(f"{name}:{model}: {e}")
    if missing:
        raise SystemExit("models not available:\n  " + "\n  ".join(missing))

    print(f"[*] {len(tasks)} answers: {len({(t.session.session.id, t.question_id) for t in tasks})} questions x "
          f"{len(configs)} configs x {len(specs)} model sets ({len(needed)} models)")
    rows = await answer_all(tasks, engines, args.concurrency)

    out = args.out or HERE / "results" / ("answers-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "answers.jsonl", rows)
    meta = {"phase": "answer", "data": str(args.data), "sessions": len(sessions), "questions": len({(r["session"], r["question"]) for r in rows}),
            "configs": [c.describe() for c in configs], "sets": [s.label for s in specs], "tier": args.tier,
            "models": [f"{e}:{m}" for e, m in needed], "embedding_model": args.embedding_model,
            "errors": sum("error" in r for r in rows)}
    (out / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] {len(rows)} answers ({meta['errors']} errors) -> {out}")
    return out


# ------------------------------------------------------------------ phase 2: judge
_THINK = re.compile(r"<think>.*?</think>", re.S)


def strip_thinking(text: str) -> str:
    text = _THINK.sub("", text or "")
    return text.split("</think>", 1)[1] if "</think>" in text else text      # an unopened block


def excerpts(session: Session, golds: list[Gold], limit: int = EXCERPT_CHARS) -> str:
    """The chunks a gold quote spans, grouped by turn: what the judge compares the answer with."""
    by_turn: dict[int, dict[int, str]] = defaultdict(dict)
    for g in golds:
        turn = session.turns[g.turn - 1]
        drafts = chunk_turn(turn.user, turn.assistant)
        ords = set(gold_ords(turn, drafts, g))
        by_turn[g.turn].update({d.ord: d.content for d in drafts if d.ord in ords})
    parts = []
    for turn in sorted(by_turn):
        text = "\n[…]\n".join(c for _, c in sorted(by_turn[turn].items()))
        parts.append(f"[turn {turn}]\n{text[:limit]}")
    return "\n\n".join(parts)


def judge_prompt(session: Session, question_id: str, answer: str) -> str:
    q = next(x for x in session.questions if x.id == question_id)
    outdated = OUTDATED_BLOCK.format(excerpts=excerpts(session, q.outdated)) if q.outdated else ""
    return JUDGE_PROMPT.format(gold=excerpts(session, q.gold), outdated=outdated, question=q.prompt.strip(),
                               answer=strip_thinking(answer).strip()[:ANSWER_CHARS] or "(empty)")


def parse_verdict(text: str) -> dict:
    """{"score": 1-5, "outdated": bool, "reason": str} from the judge's reply (a <think> block,
    a ```json fence or prose around it are fine: the last object with a score wins)."""
    text = strip_thinking(text)
    decoder = json.JSONDecoder()
    # decode from every "{" (not a brace-free regex: the reason may quote "{day} at {time}")
    for start in reversed([m.start() for m in re.finditer(r"\{", text)]):
        try:
            obj, _ = decoder.raw_decode(text, start)
        except ValueError:
            continue
        if isinstance(obj, dict) and "score" in obj:
            score = int(obj["score"])
            if 1 <= score <= 5:
                return {"score": score, "outdated": bool(obj.get("outdated", False)),
                        "reason": str(obj.get("reason", ""))}
    # invalid JSON (e.g. „quoted" text closed with a bare ") -> read the fields directly
    scores = re.findall(r'"score"\s*:\s*"?([1-5])\b', text)
    if scores:
        outdated = re.findall(r'"outdated"\s*:\s*(true|false)', text)
        reason = re.search(r'"reason"\s*:\s*"(.*)', text, re.S)
        return {"score": int(scores[-1]), "outdated": bool(outdated) and outdated[-1] == "true",
                "reason": reason.group(1).rstrip('"}` \n') if reason else ""}
    raise ValueError(f"no score in the judge's reply: {text[:200]!r}")


async def judge_all(rows: list[dict], sessions: dict[str, Session], complete, cache: JsonlCache,
                    concurrency: int = 4, retries: int = 1) -> list[dict]:
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(row: dict) -> dict:
        nonlocal done
        if "error" in row:
            return row
        prompt = judge_prompt(sessions[row["session"]], row["question"], row["answer"])
        key = JsonlCache.key(PROMPT_VERSION, prompt)
        verdict = cache.get(key)
        error = None
        for _ in range(retries + 1):
            if verdict is not None:
                break
            try:
                async with sem:
                    verdict = parse_verdict(await complete(prompt))
                cache.put(key, verdict)
            except Exception as e:
                error = f"{type(e).__name__}: {e}"
        done += 1
        if done % 100 == 0:
            print(f"    {done}/{len(rows)} judged")
        return {**row, "judge": verdict} if verdict else {**row, "judge_error": error}

    return list(await asyncio.gather(*(one(r) for r in rows)))


async def run_judge(args) -> Path:
    if "<" in args.judge_base or "<" in args.judge_key:
        raise SystemExit(f"fill in GLM_BASE_URL / GLM_API_KEY in {ENV_FILE} (or the shell env)")
    rows = _read_jsonl(args.dir / "answers.jsonl")
    run_meta = json.loads((args.dir / "run.json").read_text(encoding="utf-8")) if (args.dir / "run.json").exists() else {}
    data = args.data or Path(run_meta.get("data", DATA_DIR))       # the answer phase's dataset
    sessions = {s.id: s for s in load_dataset(data)}
    cache = JsonlCache(CACHE_DIR / f"answer-judge-{_safe(args.judge_model)}.jsonl")
    if args.insecure:
        print("[!] TLS certificate checks are off for the judge (GLM_INSECURE=true)")
    print(f"[*] judging {len(rows)} answers with {args.judge_model}")
    async with httpx.AsyncClient(verify=not args.insecure, timeout=args.timeout) as http_client:
        complete = openai_compatible(http_client, args.judge_base, args.judge_key, args.judge_model,
                                     args.temperature, args.max_tokens, json.loads(args.extra_body))
        judged = await judge_all(rows, sessions, complete, cache, args.concurrency, args.retries)
    _write_jsonl(args.dir / "judged.jsonl", judged)
    meta = run_meta
    meta.update({"judge_model": args.judge_model, "judge_prompt_version": PROMPT_VERSION,
                 "judge_errors": sum("judge_error" in r for r in judged)})
    (args.dir / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(write_report(judged, meta, args.dir / "report.md"))
    print(f"\n[OK] {args.dir}")
    return args.dir


# ------------------------------------------------------------------ report
def mcnemar_p(wins: int, losses: int) -> float:
    """Exact two-sided McNemar test on the discordant pairs."""
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _correct(r: dict) -> bool:
    return r["judge"]["score"] >= CORRECT


def write_report(rows: list[dict], meta: dict, path: Path) -> str:
    judged = [r for r in rows if "judge" in r]
    sets = list(dict.fromkeys(r["set"] for r in rows))
    configs = list(dict.fromkeys(r["config"] for r in rows))
    lines = [
        "# Answers, end to end",
        "",
        f"{meta.get('questions', '?')} questions, configs {', '.join(configs)}, model sets {', '.join(sets)}; "
        f"judge `{meta.get('judge_model', '?')}`. {len(judged)} of {len(rows)} answers judged "
        f"({sum('error' in r for r in rows)} model errors, {sum('judge_error' in r for r in rows)} judge errors). "
        f"Correct = score >= {CORRECT} of 5.",
        "",
        "## Main table",
        "",
        "*gold in ctx*: the gold reached the context (as in run.py). *correct, gold in ctx* = the model used "
        "it; *correct, gold not in ctx* = answered without it (general knowledge, a lucky guess, or the gold "
        "repeated in another turn). *old version*: decision questions answered with the replaced version.",
        "",
    ]
    header = ["set", "config", "answers", "gold in ctx", "mean score", "correct", "correct, gold in ctx",
              "correct, gold not in ctx", "old version (decision)", "cut at max tokens", "context tokens"]
    table = []
    for s in sets:
        for c in configs:
            rs = [r for r in judged if r["set"] == s and r["config"] == c]
            if not rs:
                continue
            inside = [r for r in rs if r["context"]["present"]]
            outside = [r for r in rs if not r["context"]["present"]]
            decision = [r for r in rs if r["type"] == "decision"]
            table.append([s, c, str(len(rs)), _pct(_mean(r["context"]["present"] for r in rs)),
                          _num(_mean(r["judge"]["score"] for r in rs)), _pct(_mean(_correct(r) for r in rs)),
                          _pct(_mean(_correct(r) for r in inside)), _pct(_mean(_correct(r) for r in outside)),
                          _pct(_mean(r["judge"]["outdated"] for r in decision)),
                          _pct(_mean(bool(r.get("truncated")) for r in rs)),
                          _num(_mean(r["context"]["tokens"] for r in rs), 0)])
    lines += [_table(header, table), ""]

    lines += ["## Config against config (same questions, same model set)", "",
              "*wins* of `B vs A` = questions B answers correctly and A doesn't, *losses* the other way round; "
              "p = exact McNemar (< 0.05: unlikely to be chance).", ""]
    table = []
    for s in sets:
        by_key = {(r["config"], r["session"], r["question"]): r for r in judged if r["set"] == s}
        for i, a in enumerate(configs):
            for b in configs[i + 1:]:
                pairs = [(by_key[(a, *k)], by_key[(b, *k)]) for (c, *k) in by_key if c == a and (b, *k) in by_key]
                if not pairs:
                    continue
                wins = sum(_correct(x) and not _correct(y) for x, y in pairs)
                losses = sum(_correct(y) and not _correct(x) for x, y in pairs)
                table.append([s, f"{b} vs {a}", str(len(pairs)), str(losses), str(wins),
                              _num(mcnemar_p(wins, losses), 3)])
    lines += [_table(["set", "comparison", "questions", "wins", "losses", "p"], table), ""]

    for title, key in (("question type", "type"), ("category", "category")):
        lines += [f"## Correct per {title}", ""]
        values = sorted({r[key] for r in judged})
        cols = [(s, c) for s in sets for c in configs]
        table = []
        for v in values:
            row = [v, str(len({(r["session"], r["question"]) for r in judged if r[key] == v}))]
            for s, c in cols:
                rs = [r for r in judged if r[key] == v and r["set"] == s and r["config"] == c]
                row.append(_pct(_mean(_correct(r) for r in rs)))
            table.append(row)
        lines += [_table([title, "questions"] + [f"{s} {c}" for s, c in cols], table), ""]

    experts = [r for r in judged if r["set"] == "expert"]
    if experts:
        lines += ["## Expert models", "", "The catalog model each category's questions went to.", ""]
        table = []
        for model in sorted({r["model"] for r in experts}):
            rs = [r for r in experts if r["model"] == model]
            row = [model, ", ".join(sorted({r["category"] for r in rs}))]
            row += [_pct(_mean(_correct(r) for r in rs if r["config"] == c)) for c in configs]
            row.append(_num(_mean(r["seconds"] for r in rs if r.get("seconds") is not None), 1))
            table.append(row)
        lines += [_table(["model", "categories"] + [f"correct {c}" for c in configs] + ["seconds / answer"], table), ""]

    report = "\n".join(lines)
    path.write_text(report, encoding="utf-8")
    return report


def run_report(args) -> None:
    path = args.dir / "judged.jsonl"
    if not path.exists():
        raise SystemExit(f"{path} is missing: run the judge phase first")
    meta = json.loads((args.dir / "run.json").read_text(encoding="utf-8")) if (args.dir / "run.json").exists() else {}
    print(write_report(_read_jsonl(path), meta, args.dir / "report.md"))


# ------------------------------------------------------------------ io, cli
def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="phase", required=True)

    a = sub.add_parser("answer", help="the models answer (where the models are)")
    a.add_argument("--models", nargs="+", default=["expert"], help="expert and / or <engine>:<model>")
    a.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS)
    a.add_argument("--data", type=Path, default=DATA_DIR, help="dataset folder (data/sessions/; data/multihop/ for the multi-hop set)")
    a.add_argument("--sessions", default="*.yaml", help="glob over --data")
    a.add_argument("--limit", type=int, help="only the first N questions (a smoke test)")
    a.add_argument("--catalog", type=Path, default=CATALOG, help="the expert models (as the router reads them)")
    a.add_argument("--tier", default="high", choices=["low", "high"], help="the experts' complexity tier")
    a.add_argument("--ollama", help="Ollama URL, e.g. http://127.0.0.1:11500")
    a.add_argument("--vllm", help="vLLM URL, e.g. http://127.0.0.1:8100")
    a.add_argument("--context-window", type=int, default=DEFAULT_CONTEXT_WINDOW, help="for <engine>:<model> sets")
    a.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS, help="for <engine>:<model> sets")
    a.add_argument("--concurrency", type=int, default=4, help="requests in flight per model")
    a.add_argument("--embedding-host", default="http://localhost:11434")
    a.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    a.add_argument("--out", type=Path)

    j = sub.add_parser("judge", help="the judge scores the answers (where the judge is)")
    j.add_argument("dir", type=Path, help="the folder the answer phase wrote")
    j.add_argument("--data", type=Path, help="dataset folder (default: the one in the answer phase's run.json)")
    j.add_argument("--judge-base", default=GLM_BASE_URL)
    j.add_argument("--judge-key", default=GLM_API_KEY)
    j.add_argument("--judge-model", default=GLM_MODEL)
    j.add_argument("--insecure", action=argparse.BooleanOptionalAction, default=GLM_INSECURE)
    j.add_argument("--temperature", type=float, default=0.0)
    j.add_argument("--max-tokens", type=int, default=4096, help="room for a thinking model's reasoning too")
    j.add_argument("--extra-body", default="{}", help="JSON merged into each request, server-specific options")
    j.add_argument("--timeout", type=float, default=300)
    j.add_argument("--concurrency", type=int, default=4)
    j.add_argument("--retries", type=int, default=1)

    r = sub.add_parser("report", help="the tables again from judged.jsonl")
    r.add_argument("dir", type=Path)

    args = ap.parse_args()
    if args.phase == "answer":
        asyncio.run(run_answer(args))
    elif args.phase == "judge":
        asyncio.run(run_judge(args))
    else:
        run_report(args)


if __name__ == "__main__":
    main()
