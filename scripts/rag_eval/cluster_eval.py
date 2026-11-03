"""
End-to-end evaluation in the cluster (docs/rag-planv2.md 6, step 11): the dataset's questions sent
through the gateway, with a RAG variant per session, to check that the cluster agrees with the
offline harness and to measure the latencies per stage.

The conversations are NOT replayed through the models (the answers would change and the gold
quotes would no longer exist). Instead:
  1. seed: each dataset conversation is written to Postgres as it is (turns + routing state) and
     indexed with memory/indexer.py -- the same code as the memory service
  2. per (question, variant): the seeded session is cloned in SQL (turns + chunks, no
     re-embedding), so every question starts from the same history, and the question goes to
     POST /generate with that session id and `variant`
  3. the question's stored turn says what reached the model: recent_turns (whole turns) and
     retrieved_chunk_ids (fragments); the reply carries the stage timings
  4. optionally an LLM judge scores each answer 1-5 against the gold quote

Needs: the router with ALLOW_VARIANT_OVERRIDE=true (router.rag.allowVariantOverride), the
gateway and the embedding model reachable, and the Postgres credentials (dbctl.sh exec):

    kubectl -n moe-router port-forward svc/moe-gateway 8000:8000 &
    kubectl -n moe-router port-forward svc/moe-ollama 11435:11434 &
    ./src/database/dbctl.sh exec .venv/bin/python scripts/rag_eval/cluster_eval.py \\
        --gateway http://localhost:8000 --embedding-host http://localhost:11435 \\
        --variants V0 V1 V2 --sessions 's01*'

Every session it creates is deleted at the end (--keep to inspect them). Writes
results/cluster-<timestamp>/{answers.jsonl, report.md}.
"""
import argparse
import asyncio
import datetime
import json
import re
import sys
import time
import uuid
from collections import defaultdict
from pathlib import Path

import httpx

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from dataset import DATA_DIR, Question, Session, check, gold_ords, load_dataset
from report import _mean, _num, _pct, _table
from sqlalchemy import insert, literal, select

from database import ChatSessionRepository, ChunkRepository, TurnRepository, dispose_engine, get_session
from database.models.chunk.schema import Chunk
from database.models.turn.schema import Turn
from memory.indexer import index_turn
from shared.memory.chunking import chunk_turn
from shared.memory.embedding import OllamaEmbeddingClient

JUDGE_PROMPT = """You grade an assistant's answer to a question about an earlier part of a conversation.
The correct answer contains this information (quoted from the conversation):
{gold}

Question: {question}

Answer: {answer}

Score 1-5: 5 = states the quoted information correctly, 3 = partly / vaguely, 1 = wrong, missing or
invented. Reply with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}"""


# ------------------------------------------------------------------ seeding
async def seed(session: Session, embedder) -> uuid.UUID:
    """The conversation as stored turns, indexed. Returns its (template) session id."""
    sid = uuid.uuid4()
    async with get_session() as db:
        await ChatSessionRepository(db).ensure(sid)
        turns = TurnRepository(db)
        ids = [await turns.create(sid, n, t.user, t.assistant, effective_domain=session.category,
                                  rag_variant="seed")
               for n, t in enumerate(session.turns, start=1)]
        await ChatSessionRepository(db).update_routing_state(sid, session.category, 0.9, 0.5)
    for turn_id in ids:
        await index_turn(turn_id, embedder)
    return sid


def _copy_columns(table, skip):
    return [c for c in table.columns if c.name not in skip]


async def clone(source: uuid.UUID) -> uuid.UUID:
    """A copy of a seeded session (routing state, turns, chunks with their vectors) under a new id."""
    new = uuid.uuid4()
    async with get_session() as db:
        state = await ChatSessionRepository(db).get(source)
        await ChatSessionRepository(db).ensure(new)
        await ChatSessionRepository(db).update_routing_state(new, state.domain, state.domain_confidence,
                                                             state.complexity_score)
        turn_cols = _copy_columns(Turn.__table__, {"id", "session_id", "created_at"})
        await db.execute(insert(Turn).from_select(
            ["session_id"] + [c.name for c in turn_cols],
            select(literal(new, Turn.session_id.type), *turn_cols).where(Turn.session_id == source)))
        chunk_cols = _copy_columns(Chunk.__table__, {"id", "session_id", "turn_id", "created_at"})
        target = Turn.__table__.alias("target")
        await db.execute(insert(Chunk).from_select(
            ["session_id", "turn_id"] + [c.name for c in chunk_cols],
            select(literal(new, Chunk.session_id.type), target.c.id, *chunk_cols)
            .join(target, (target.c.session_id == new) & (target.c.turn == Chunk.turn))
            .where(Chunk.session_id == source)))
    return new


async def delete_sessions(ids) -> None:
    async with get_session() as db:
        repo = ChatSessionRepository(db)
        for sid in ids:
            await repo.delete(sid)


# ------------------------------------------------------------------ asking
async def gold_chunk_ids(sid: uuid.UUID, session: Session, q: Question) -> set[int]:
    async with get_session() as db:
        turns = await TurnRepository(db).list_recent(sid, len(session.turns) + 1)
        by_turn = {t.turn: await ChunkRepository(db).list_for_turn(t.id) for t in turns}
    ids = set()
    for g in q.gold:
        t = session.turns[g.turn - 1]
        ords = set(gold_ords(t, chunk_turn(t.user, t.assistant), g))
        ids |= {c.id for c in by_turn.get(g.turn, []) if c.ord in ords}
    return ids


async def ask(http: httpx.AsyncClient, gateway: str, session: Session, q: Question, template: uuid.UUID,
              variant: str) -> dict:
    sid = await clone(template)
    gold = await gold_chunk_ids(sid, session, q)
    t0 = time.perf_counter()
    r = await http.post(f"{gateway}/generate", json={"prompt": q.prompt, "session_id": str(sid),
                                                      "variant": variant})
    total_ms = round((time.perf_counter() - t0) * 1000, 1)
    row = {"session": session.id, "category": session.category, "question": q.id, "type": q.type,
           "variant": variant, "session_id": str(sid), "status": r.status_code, "total_ms": total_ms}
    if r.status_code != 200:
        row["error"] = r.text[:300]
        return row
    reply = r.json()
    routing = reply.get("routing") or {}
    n = len(session.turns)
    whole = set(range(n - (routing.get("recent_turns") or 0) + 1, n + 1))
    fragments = set(routing.get("retrieved_chunk_ids") or [])
    gold_turns = {g.turn for g in q.gold}
    via = "whole" if gold_turns & whole else "fragment" if gold & fragments else "none"
    row.update({
        "answer": reply.get("response", ""), "model": reply.get("model"),
        "rag_variant": routing.get("rag_variant"), "present": via != "none", "via": via,
        "recent_turns": routing.get("recent_turns"), "fragments": len(fragments),
        "context_tokens": routing.get("context_tokens"), "prompt_tokens": reply.get("prompt_tokens"),
        "classify_ms": routing.get("classify_ms"), "generate_ms": reply.get("latency_ms"),
        "retrieval": routing.get("retrieval"), "gold_quotes": [g.quote for g in q.gold],
        "prompt": q.prompt,
    })
    return row


async def judge(http: httpx.AsyncClient, base: str, model: str, row: dict) -> None:
    prompt = JUDGE_PROMPT.format(gold="\n".join(f"- {g}" for g in row["gold_quotes"]),
                                 question=row["prompt"].strip(), answer=row["answer"].strip())
    r = await http.post(f"{base}/v1/chat/completions", json={
        "model": model, "temperature": 0, "messages": [{"role": "user", "content": prompt}]})
    r.raise_for_status()
    text = r.json()["choices"][0]["message"]["content"]
    match = re.search(r"\{.*\}", text, re.S)
    try:
        verdict = json.loads(match.group(0)) if match else {}
        row["judge"] = int(verdict["score"])
        row["judge_reason"] = verdict.get("reason", "")
    except (ValueError, KeyError, TypeError):
        row["judge_error"] = text[:200]


# ------------------------------------------------------------------ report
def report(rows: list[dict], variants: list[str]) -> str:
    ok = [r for r in rows if r.get("status") == 200]
    by_variant = defaultdict(list)
    for r in ok:
        by_variant[r["variant"]].append(r)

    def stage(rs, key):
        return _num(_mean(r["retrieval"][key] for r in rs if r.get("retrieval") and key in r["retrieval"]), 0)

    header = ["variant", "answers", "ran as", "in context", "via fragment", "judge (1-5)", "context tokens",
              "classify ms", "embed ms", "search ms", "fetch ms", "generate ms", "total ms"]
    table = []
    for v in variants:
        rs = by_variant.get(v, [])
        ran = ",".join(sorted({str(r.get("rag_variant")) for r in rs})) or "—"
        table.append([
            v, str(len(rs)), ran, _pct(_mean(r["present"] for r in rs)),
            _pct(_mean(r["via"] == "fragment" for r in rs)),
            _num(_mean(r["judge"] for r in rs if "judge" in r)),
            _num(_mean(r["context_tokens"] for r in rs if r.get("context_tokens") is not None), 0),
            _num(_mean(r["classify_ms"] for r in rs if r.get("classify_ms") is not None), 0),
            stage(rs, "embed_ms"), stage(rs, "search_ms"), stage(rs, "fetch_ms"),
            _num(_mean(r["generate_ms"] for r in rs if r.get("generate_ms") is not None), 0),
            _num(_mean(r["total_ms"] for r in rs), 0),
        ])
    failed = [r for r in rows if r.get("status") != 200]
    out = ["# RAG evaluation in the cluster", "", _table(header, table), ""]
    if failed:
        out += [f"{len(failed)} request(s) failed, e.g. {failed[0].get('status')}: {failed[0].get('error', '')[:200]}", ""]
    return "\n".join(out)


# ------------------------------------------------------------------ main
async def run(args) -> None:
    sessions = load_dataset(DATA_DIR, args.sessions)
    problems = [p for s in sessions for p in check(s)]
    if problems:
        raise SystemExit("dataset problems:\n  " + "\n  ".join(problems))
    embedder = OllamaEmbeddingClient(args.embedding_host, args.embedding_model, timeout=600)
    out = HERE / "results" / ("cluster-" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)

    created: list[uuid.UUID] = []
    rows: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=args.timeout) as http:
            for s in sessions:
                print(f"[*] seeding {s.id} ({len(s.turns)} turns)")
                template = await seed(s, embedder)
                created.append(template)
                for q in s.questions:
                    for v in args.variants:
                        row = await ask(http, args.gateway, s, q, template, v)
                        created.append(uuid.UUID(row["session_id"]))
                        if args.judge_model and row.get("status") == 200:
                            await judge(http, args.judge_base, args.judge_model, row)
                        rows.append(row)
                        print(f"    {q.id} {v}: {row.get('status')} ran={row.get('rag_variant')} "
                              f"via={row.get('via')} judge={row.get('judge', '-')} {row['total_ms']:.0f} ms")
                        with (out / "answers.jsonl").open("a", encoding="utf-8") as f:
                            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        if not args.keep:
            await delete_sessions(created)
        await dispose_engine()

    text = report(rows, args.variants)
    (out / "report.md").write_text(text, encoding="utf-8")
    print(text)
    print(f"[OK] {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gateway", default="http://localhost:8000")
    ap.add_argument("--embedding-host", default="http://localhost:11434")
    ap.add_argument("--embedding-model", default="bge-m3")
    ap.add_argument("--variants", nargs="+", default=["V0", "V1", "V2"])
    ap.add_argument("--sessions", default="*.yaml")
    ap.add_argument("--timeout", type=float, default=600)
    ap.add_argument("--judge-base", default="http://localhost:11434", help="an OpenAI-compatible server")
    ap.add_argument("--judge-model", help="e.g. qwen2.5:7b; omitted = no judge")
    ap.add_argument("--keep", action="store_true", help="don't delete the sessions created")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
