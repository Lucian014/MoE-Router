"""
Retrieval graded by an LLM judge (GLM, or any OpenAI-compatible server).

run.py checks the retrieved chunks against the gold labels written with the dataset: a chunk counts
only if it overlaps the quoted answer. This script asks a second opinion: the judge reads every chunk
the search returned (and every gold chunk) next to the question and grades it on its own:

    2 = contains the information that answers the question, in its latest version
    1 = related, not the answer: same topic, only part of it, or a version changed later on
    0 = unrelated

So it measures what the gold labels cannot (how much of the top-k is noise, answers repeated in
turns the gold doesn't list) and it checks the gold labels themselves (gold chunks the judge gives
0, chunks it gives 2 that no gold lists: review those by hand).

    # the judge: cp scripts/rag_eval/.env.example scripts/rag_eval/.env, fill in GLM_BASE_URL /
    # GLM_API_KEY / GLM_MODEL (the shell env wins over the file)
    python scripts/rag_eval/judge_retrieval.py --embedding-host http://127.0.0.1:11500
    python scripts/rag_eval/judge_retrieval.py --sessions 's1[4-6]*' --top-k 10
    python scripts/rag_eval/judge_retrieval.py --extra-body '{"thinking": {"type": "disabled"}}'

Metrics per config (mean over questions):
    gold recall@k   any top-k chunk overlaps a gold quote (the same number as run.py)
    judge hit@k     any top-k chunk the judge grades 2
    P@k (=2, >=1)   the share of the top-k graded 2 / at least 1: how much of the context is useful
    nDCG@k          graded ranking quality, ideal order = every chunk judged for that question

Writes results/judge-<timestamp>/{judged.jsonl, metrics.jsonl, report.md}. Grades are cached in
.cache/judge-<model>.jsonl per (question, chunk text), so a rerun only asks about chunks it hasn't
seen; delete that file to grade again from scratch.
"""
import argparse
import asyncio
import datetime
import hashlib
import json
import math
import os
import re
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import DATA_DIR, Question, check, load_dataset
from harness import (CACHE_DIR, CachedEmbedder, Config, IndexedChunk, IndexedSession, corpus_stats,
                     index_sessions, main_configs, query_vectors, retrieve)
from shared.config import DEFAULT_EMBEDDING_MODEL, RetrievalSettings
from shared.memory.embedding import OllamaEmbeddingClient

# ------------------------------------------------------------------ the judge: fill these in
ENV_FILE = HERE / ".env"         # git-ignored; copy .env.example and fill it in


def load_env_file(path: Path = ENV_FILE) -> None:
    """KEY=value lines into os.environ (blank lines and # comments skipped, optional quotes stripped).
    A variable already set in the shell wins over the file."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.removeprefix("export ").strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        os.environ.setdefault(key, value)


load_env_file()
# The key lives in scripts/rag_eval/.env (or the shell env), never in this file.
GLM_BASE_URL = os.environ.get("GLM_BASE_URL", "https://<GLM_HOST>/v1")   # OpenAI-compatible base URL
GLM_API_KEY = os.environ.get("GLM_API_KEY", "<GLM_API_KEY>")             # the API key
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-5.3")                       # the model name the server expects
# true = the httpx client skips TLS certificate checks (a self-signed / internal certificate). Insecure.
GLM_INSECURE = os.environ.get("GLM_INSECURE", "true").strip().lower() in ("1", "true", "yes")

PROMPT_VERSION = 2           # bump when JUDGE_PROMPT changes: the cached grades no longer apply
OUTLINE_CHARS = 300          # per user message in the conversation outline
EXCERPT_CHARS = 2500         # per chunk (chunks are <= 400 tokens, this rarely cuts)

JUDGE_PROMPT = """You check the search results of a chat assistant's memory.

A user and an assistant had the conversation outlined below (only the user's messages, shortened). \
After the last turn the user asks a new question. A search over the earlier turns returned the \
excerpts listed after it. Grade each excerpt on its own:

2 = it contains the information that answers the question, in its latest version
1 = related, but not the answer: the same topic, only part of the answer, or a version that the \
conversation changed later on (use the outline to tell)
0 = unrelated to the question

Conversation outline (user messages, turn by turn):
{outline}

Question (asked after turn {last}):
{question}

Excerpts:
{excerpts}

Reply with JSON only, one entry per excerpt:
{{"grades": [{{"id": <excerpt id>, "grade": <0, 1 or 2>, "reason": "<a few words>"}}]}}"""


# ------------------------------------------------------------------ talking to the judge
Complete = Callable[[str], Awaitable[str]]


def openai_compatible(http_client: httpx.AsyncClient, base_url: str, api_key: str, model: str,
                      temperature: float, max_tokens: int, extra_body: dict) -> Complete:
    """One user message in, the reply text out: POST {base_url}/chat/completions."""
    url = base_url.rstrip("/") + "/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}"}

    async def complete(prompt: str) -> str:
        r = await http_client.post(url, headers=headers, json={
            "model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}], **extra_body})
        r.raise_for_status()
        choice = r.json()["choices"][0]
        text = choice["message"].get("content") or ""
        if not text.strip() and choice.get("finish_reason") == "length":
            raise RuntimeError("empty reply cut at max_tokens (a thinking model?): raise --max-tokens")
        return text

    return complete


_THINK = re.compile(r"<think>.*?</think>", re.S)


def parse_grades(text: str) -> dict[int, tuple[int, str]]:
    """{excerpt id: (grade, reason)} from the judge's reply. Tolerates a <think> block, a ```json
    fence and prose around the JSON; takes the last object that has "grades"."""
    text = _THINK.sub("", text)
    decoder, found = json.JSONDecoder(), None
    for i, ch in enumerate(text):
        if ch != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except ValueError:
            continue
        if isinstance(obj, dict) and isinstance(obj.get("grades"), list):
            found = obj
    out = {}
    for g in (found or {}).get("grades", []):
        try:
            grade = int(g["grade"])
            if grade in (0, 1, 2):
                out[int(g["id"])] = (grade, str(g.get("reason", "")))
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _flat(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rstrip() + " …"


def outline(s: IndexedSession) -> str:
    return "\n".join(f"{n}. {_flat(t.user, OUTLINE_CHARS)}" for n, t in enumerate(s.session.turns, start=1))


def excerpt_head(c: IndexedChunk) -> str:
    """What the model sees above a fragment (assembly.py): the turn and the question it answered."""
    who = "the user's message" if c.source == "prompt" else "the assistant's answer"
    if not c.header:
        return f"(turn {c.turn}, from {who})"
    label = "topic" if c.header_from_answer else "the user had asked"
    return f"(turn {c.turn}, from {who}; {label}: {_flat(c.header, OUTLINE_CHARS)})"


def judge_prompt(s: IndexedSession, q: Question, chunks: list[IndexedChunk]) -> str:
    excerpts = "\n\n".join(f"[id {i}] {excerpt_head(c)}\n{c.content.strip()[:EXCERPT_CHARS]}"
                            for i, c in enumerate(chunks, start=1))
    return JUDGE_PROMPT.format(outline=outline(s), last=len(s.session.turns), question=q.prompt.strip(),
                               excerpts=excerpts)


# ------------------------------------------------------------------ cached grades
class GradeCache:
    """(model, prompt version, session, question, chunk text) -> (grade, reason), in a jsonl file."""

    def __init__(self, model: str, cache_dir: Path | None = CACHE_DIR):
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", model)
        self.model = model
        self._path = cache_dir / f"judge-{safe}.jsonl" if cache_dir else None
        self._grades: dict[str, tuple[int, str]] = {}
        if self._path and self._path.exists():
            for line in self._path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                self._grades[row["k"]] = (row["grade"], row["reason"])

    def key(self, s: IndexedSession, q: Question, c: IndexedChunk) -> str:
        raw = json.dumps([self.model, PROMPT_VERSION, s.session.id, q.prompt, c.content], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def get(self, key: str) -> tuple[int, str] | None:
        return self._grades.get(key)

    def put(self, items: dict[str, tuple[int, str]]) -> None:
        self._grades.update(items)
        if self._path and items:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as f:
                for k, (grade, reason) in items.items():
                    f.write(json.dumps({"k": k, "grade": grade, "reason": reason}, ensure_ascii=False) + "\n")


async def grade_question(s: IndexedSession, q: Question, chunk_ids: list[int], complete: Complete,
                         cache: GradeCache, batch: int, retries: int) -> dict[int, tuple[int, str]]:
    """{chunk id: (grade, reason)} for `chunk_ids`; a chunk the judge never graded is left out."""
    by_id = {c.id: c for c in s.chunks}
    keys = {cid: cache.key(s, q, by_id[cid]) for cid in chunk_ids}
    out = {cid: g for cid in chunk_ids if (g := cache.get(keys[cid])) is not None}
    todo = [cid for cid in chunk_ids if cid not in out]
    for start in range(0, len(todo), batch):
        pending = todo[start:start + batch]
        for _ in range(1 + retries):
            if not pending:
                break
            chunks = [by_id[cid] for cid in pending]
            try:
                grades = parse_grades(await complete(judge_prompt(s, q, chunks)))
            except (httpx.HTTPError, RuntimeError, KeyError, IndexError, ValueError) as e:
                print(f"    [!] {s.session.id}/{q.id}: {type(e).__name__}: {e}")
                continue
            got = {cid: grades[i] for i, cid in enumerate(pending, start=1) if i in grades}
            cache.put({keys[cid]: g for cid, g in got.items()})
            out.update(got)
            pending = [cid for cid in pending if cid not in got]
        if pending:
            print(f"    [!] {s.session.id}/{q.id}: no grade for {len(pending)} chunk(s)")
    return out


# ------------------------------------------------------------------ metrics
def ndcg(grades: list[int], ideal: list[int], k: int) -> float:
    def dcg(gs):
        return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(gs[:k]))
    best = dcg(sorted(ideal, reverse=True))
    return dcg(grades) / best if best else 0.0


def question_metrics(top: list[int], grades: dict[int, tuple[int, str]], gold: set[int]) -> dict:
    """Unjudged chunks count as 0 (and are counted, so a broken judge shows up in the report)."""
    g = [grades[cid][0] if cid in grades else 0 for cid in top]
    n = len(top) or 1
    return {
        "gold_recall": any(cid in gold for cid in top),
        "judge_hit": 2 in g,
        "p_relevant": sum(x == 2 for x in g) / n,
        "p_related": sum(x >= 1 for x in g) / n,
        "ndcg": ndcg(g, [x for x, _ in grades.values()], len(top)),
        "unjudged": sum(cid not in grades for cid in top),
    }


# ------------------------------------------------------------------ report
def _pct(xs) -> str:
    xs = list(xs)
    return f"{100 * sum(xs) / len(xs):.0f}%" if xs else "—"


def _mean(xs) -> str:
    xs = list(xs)
    return f"{sum(xs) / len(xs):.2f}" if xs else "—"


def write_report(metrics: list[dict], judged: list[dict], meta: dict, path: Path) -> str:
    configs = list(dict.fromkeys(m["config"] for m in metrics))
    lines = [
        "# Retrieval graded by an LLM judge", "",
        f"{meta['sessions']} sessions, {meta['questions']} questions, {meta['judged']} (question, chunk) "
        f"pairs graded by `{meta['judge_model']}`. Grades: 2 = answers it, 1 = related / partial / "
        "changed later, 0 = unrelated.", "",
        "## Main table", "",
        "| config | gold recall@k | judge hit@k | P@k (=2) | P@k (≥1) | nDCG@k | unjudged in top-k |",
        "|---|---|---|---|---|---|---|",
    ]
    for cfg in configs:
        ms = [m for m in metrics if m["config"] == cfg]
        lines.append(f"| {cfg} | {_pct(m['gold_recall'] for m in ms)} | {_pct(m['judge_hit'] for m in ms)} | "
                     f"{_mean(m['p_relevant'] for m in ms)} | {_mean(m['p_related'] for m in ms)} | "
                     f"{_mean(m['ndcg'] for m in ms)} | {sum(m['unjudged'] for m in ms)} |")

    types = sorted({m["type"] for m in metrics})
    lines += ["", "## Per question type (gold recall / judge hit)", "",
              "| question type | questions | " + " | ".join(configs) + " |",
              "|---|---|" + "---|" * len(configs)]
    for t in types:
        cells = []
        for cfg in configs:
            ms = [m for m in metrics if m["config"] == cfg and m["type"] == t]
            cells.append(f"{_pct(m['gold_recall'] for m in ms)} / {_pct(m['judge_hit'] for m in ms)}")
        n = len({(m["session"], m["question"]) for m in metrics if m["type"] == t})
        lines.append(f"| {t} | {n} | " + " | ".join(cells) + " |")

    gold_rows = [j for j in judged if j["gold"]]
    lines += ["", "## Judge vs gold labels", "",
              f"Gold chunks graded 2: {_pct(j['grade'] == 2 for j in gold_rows)}, "
              f"1: {_pct(j['grade'] == 1 for j in gold_rows)}, 0: {_pct(j['grade'] == 0 for j in gold_rows)} "
              f"({len(gold_rows)} gold chunks; a gold quote can span chunks, so a 1 is not always wrong).",
              f"Outdated chunks graded 2 (the judge missed the change): "
              f"{sum(j['grade'] == 2 for j in judged if j['outdated'])} of {sum(j['outdated'] for j in judged)}.",
              "", "### Gold chunks graded 0 (a wrong gold label, or a judge mistake)", ""]
    lines += [f"- {j['session']}/{j['question']} turn {j['turn']}: {j['reason']}"
              for j in gold_rows if j["grade"] == 0] or ["- none"]
    lines += ["", "### Chunks graded 2 outside the gold turns (a missing gold turn? review by hand)", ""]
    lines += [f"- {j['session']}/{j['question']} turn {j['turn']} ({j['source']}): {j['reason']}"
              for j in judged if j["grade"] == 2 and not j["gold"] and not j["gold_turn"]] or ["- none"]

    report = "\n".join(lines) + "\n"
    path.write_text(report, encoding="utf-8")
    return report


# ------------------------------------------------------------------ run
async def judge_retrieval(indexed: list[IndexedSession], configs: list[Config], embedder: CachedEmbedder,
                          complete: Complete, cache: GradeCache, batch: int = 10, retries: int = 1,
                          concurrency: int = 4,
                          settings: RetrievalSettings | None = None) -> tuple[list[dict], list[dict]]:
    """(metrics rows: one per question x config, judged rows: one per graded (question, chunk))."""
    settings = settings or RetrievalSettings()
    vectors = await query_vectors(indexed, embedder)
    configs = [c for c in configs if c.preset.retrieves]
    corpus = corpus_stats(indexed)
    sem = asyncio.Semaphore(concurrency)

    async def one(s: IndexedSession, q: Question) -> tuple[list[dict], list[dict]]:
        tops = {cfg.name: retrieve(s, q, cfg, settings, vectors[(q.id, s.session.id, cfg.dense_query)],
                               corpus)[:cfg.top_k]
                for cfg in configs}
        gold, outdated = s.gold_ids[q.id], s.outdated_ids[q.id]
        gold_turns = {g.turn for g in q.gold}
        to_grade = list(dict.fromkeys([cid for top in tops.values() for cid in top] + sorted(gold)))
        async with sem:
            grades = await grade_question(s, q, to_grade, complete, cache, batch, retries)
        print(f"[*] {s.session.id}/{q.id}: {len(grades)}/{len(to_grade)} chunks graded")
        base = {"session": s.session.id, "question": q.id, "type": q.type, "lang": q.lang}
        metrics = [{**base, "config": name, **question_metrics(top, grades, gold)} for name, top in tops.items()]
        by_id = {c.id: c for c in s.chunks}
        judged = [{**base, "chunk": cid, "turn": by_id[cid].turn, "source": by_id[cid].source,
                   "gold": cid in gold, "gold_turn": by_id[cid].turn in gold_turns, "outdated": cid in outdated,
                   "grade": grade, "reason": reason}
                  for cid, (grade, reason) in grades.items()]
        return metrics, judged

    results = await asyncio.gather(*(one(s, q) for s in indexed for q in s.session.questions))
    return [m for ms, _ in results for m in ms], [j for _, js in results for j in js]


async def run(args) -> Path:
    if "<" in args.judge_base or "<" in args.judge_key:
        raise SystemExit(f"fill in GLM_BASE_URL / GLM_API_KEY in {ENV_FILE} (or the shell env)")
    sessions = load_dataset(args.data, args.sessions)
    if not sessions:
        raise SystemExit(f"no sessions match {args.sessions!r} in {args.data}")
    ready = []
    for s in sessions:                   # an unfinished session (generate.py, gold) is skipped, not fatal
        problems = check(s) + ([] if s.questions else [f"{s.id}: no questions yet"])
        if problems:
            print(f"[!] skipping {s.id}: " + "; ".join(problems[:3]) + (" ..." if len(problems) > 3 else ""))
        else:
            ready.append(s)
    if not ready:
        raise SystemExit("no session is ready (python scripts/rag_eval/dataset.py lists the problems)")
    sessions = ready

    embedder = CachedEmbedder(OllamaEmbeddingClient(args.embedding_host, args.embedding_model, timeout=600))
    configs = [c for c in main_configs(args.top_k) if c.preset.name in args.presets]
    indexed = await index_sessions(sessions, embedder)
    cache = GradeCache(args.judge_model)

    if args.insecure:
        print("[!] TLS certificate checks are off for the judge (GLM_INSECURE=true)")
    async with httpx.AsyncClient(verify=not args.insecure, timeout=args.timeout) as http_client:
        complete = openai_compatible(http_client, args.judge_base, args.judge_key, args.judge_model,
                                     args.temperature, args.max_tokens, json.loads(args.extra_body))
        metrics, judged = await judge_retrieval(indexed, configs, embedder, complete, cache, args.batch,
                                                args.retries, args.concurrency)

    out = args.out or HERE / "results" / ("judge-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("metrics.jsonl", metrics), ("judged.jsonl", judged)):
        with (out / name).open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
    meta = {"sessions": len(sessions), "questions": sum(len(s.questions) for s in sessions),
            "judged": len(judged), "judge_model": args.judge_model,
            "configs": [c.describe() for c in configs if c.preset.retrieves]}
    (out / "run.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(write_report(metrics, judged, meta, out / "report.md"))
    print(f"\n[OK] {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judge-base", default=GLM_BASE_URL, help="OpenAI-compatible base URL, e.g. https://host/v1")
    ap.add_argument("--judge-key", default=GLM_API_KEY)
    ap.add_argument("--judge-model", default=GLM_MODEL)
    ap.add_argument("--insecure", action=argparse.BooleanOptionalAction, default=GLM_INSECURE,
                    help="skip TLS certificate checks (default: GLM_INSECURE from .env, true)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=4096, help="room for a thinking model's reasoning too")
    ap.add_argument("--extra-body", default="{}", help="JSON merged into each request, server-specific options")
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--concurrency", type=int, default=4, help="questions graded at the same time")
    ap.add_argument("--batch", type=int, default=10, help="chunks per judge call")
    ap.add_argument("--retries", type=int, default=1, help="extra calls for chunks the reply left out")
    ap.add_argument("--embedding-host", default="http://localhost:11434")
    ap.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    ap.add_argument("--data", type=Path, default=DATA_DIR, help="dataset folder (data/sessions/; data/multihop/ for the multi-hop set)")
    ap.add_argument("--sessions", default="*.yaml", help="glob in --data")
    ap.add_argument("--presets", nargs="+", default=["V1", "V2"])
    ap.add_argument("--top-k", type=int, nargs="+", default=[5, 10])
    ap.add_argument("--out", type=Path)
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
