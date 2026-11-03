"""
The evaluation set (docs/rag-planv2.md 6): conversations + questions about their OLD turns, with
gold labels as (turn, quote) -- never chunk ids, so the labels survive a new chunker version.

One YAML file per conversation in data/sessions/ (YAML so code and long texts stay readable for the
manual check):

    id: s01-fastapi-jwt
    category: coding                 # catalog category, for the per-category tables
    lang: ro
    turns:                           # turn 1, 2, ... in order
      - user: |
          ...
        assistant: |
          ...
    questions:                       # each asked as turn len(turns) + 1, independently
      - id: q1
        type: semantic               # semantic | exact | ambiguous | code-pasted | error-only
        lang: ro                     # optional, defaults to the session's
        prompt: |
          ...
        gold:                        # the answer is in (one of) these
          - {turn: 3, quote: "create_access_token"}
        outdated:                    # optional: an older version that must NOT be what comes back
          - {turn: 2, quote: "..."}

A quote must appear verbatim in that turn's prompt or answer, inside a part that becomes a chunk
(the answer; or code, a log, a long text pasted in the prompt). Check the files with:

    python scripts/rag_eval/dataset.py            # every file in data/sessions/
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from shared.memory.chunking import ChunkDraft, chunk_turn

DATA_DIR = Path(__file__).resolve().parent / "data" / "sessions"
QUESTION_TYPES = ("semantic", "exact", "ambiguous", "code-pasted", "error-only")


@dataclass(frozen=True)
class Turn:
    user: str
    assistant: str


@dataclass(frozen=True)
class Gold:
    turn: int
    quote: str


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    lang: str
    prompt: str
    gold: list[Gold]
    outdated: list[Gold] = field(default_factory=list)


@dataclass(frozen=True)
class Session:
    id: str
    category: str
    lang: str
    turns: list[Turn]
    questions: list[Question]

    @property
    def next_turn(self) -> int:
        return len(self.turns) + 1


def _golds(items) -> list[Gold]:
    return [Gold(int(g["turn"]), str(g["quote"])) for g in items or []]


def load_session(path: Path) -> Session:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    lang = raw.get("lang", "ro")
    return Session(
        id=raw["id"], category=raw["category"], lang=lang,
        turns=[Turn(t["user"], t["assistant"]) for t in raw["turns"]],
        questions=[Question(id=q["id"], type=q["type"], lang=q.get("lang", lang), prompt=q["prompt"],
                            gold=_golds(q["gold"]), outdated=_golds(q.get("outdated")))
                   for q in raw.get("questions", [])],
    )


def load_dataset(directory: Path = DATA_DIR, pattern: str = "*.yaml") -> list[Session]:
    return [load_session(p) for p in sorted(directory.glob(pattern))]


# ------------------------------------------------------------------ quotes -> chunks
def chunk_spans(turn: Turn, drafts: list[ChunkDraft]) -> list[tuple[str, int, int]]:
    """(source, start, end) of every chunk in its turn's prompt / answer. The answer's chunks join
    back to it exactly; a prompt's chunks are slices of it in order (its short prose is skipped)."""
    spans, cursor = [], {"prompt": 0, "response": 0}
    for d in drafts:
        text = turn.user if d.source == "prompt" else turn.assistant
        start = text.find(d.content, cursor[d.source])
        if start < 0:
            raise ValueError(f"chunk {d.ord} is not a slice of its {d.source}")
        spans.append((d.source, start, start + len(d.content)))
        cursor[d.source] = start + len(d.content)
    return spans


def locate(turn: Turn, quote: str) -> tuple[str, int, int] | None:
    for source, text in (("response", turn.assistant), ("prompt", turn.user)):
        start = text.find(quote)
        if start >= 0:
            return source, start, start + len(quote)
    return None


def gold_ords(turn: Turn, drafts: list[ChunkDraft], gold: Gold) -> list[int]:
    """The ords of the chunks of `turn` that overlap the quote."""
    where = locate(turn, gold.quote)
    if where is None:
        return []
    source, start, end = where
    return [d.ord for d, (s, a, b) in zip(drafts, chunk_spans(turn, drafts))
            if s == source and a < end and start < b]


def check(session: Session) -> list[str]:
    problems = []
    drafts = [chunk_turn(t.user, t.assistant) for t in session.turns]
    ids = set()
    for q in session.questions:
        where = f"{session.id}/{q.id}"
        if q.id in ids:
            problems.append(f"{where}: duplicate question id")
        ids.add(q.id)
        if q.type not in QUESTION_TYPES:
            problems.append(f"{where}: unknown type {q.type!r}")
        if not q.gold:
            problems.append(f"{where}: no gold")
        for kind, golds in (("gold", q.gold), ("outdated", q.outdated)):
            for g in golds:
                if not 1 <= g.turn <= len(session.turns):
                    problems.append(f"{where}: {kind} turn {g.turn} out of range")
                    continue
                turn = session.turns[g.turn - 1]
                if locate(turn, g.quote) is None:
                    problems.append(f"{where}: {kind} quote not in turn {g.turn}: {g.quote[:60]!r}")
                elif not gold_ords(turn, drafts[g.turn - 1], g):
                    problems.append(f"{where}: {kind} quote of turn {g.turn} is in no chunk "
                                    f"(short prompt prose is the header, not a chunk): {g.quote[:60]!r}")
    return problems


def main() -> int:
    sessions = load_dataset()
    problems = [p for s in sessions for p in check(s)]
    turns = sum(len(s.turns) for s in sessions)
    questions = sum(len(s.questions) for s in sessions)
    print(f"{len(sessions)} sessions, {turns} turns, {questions} questions")
    for s in sessions:
        print(f"  {s.id:32s} {s.category:17s} {s.lang}  {len(s.turns):3d} turns  {len(s.questions):2d} questions")
    for p in problems:
        print(f"[!] {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
