"""Chunking a turn (docs/rag-planv2.md 4.3): segment -> split what is too long -> merge what is too
small. The constants live here and only here; changing any of them changes CHUNKER_VERSION, so rows
produced by different rules are never mixed (chunks.chunker_version).

What becomes chunks:
    the answer                                   always              source = "response"
    code, logs, formulas, tables in the prompt   always              source = "prompt"
    the prompt's prose                           only if it is long  source = "prompt"
                                                 (an article to summarize / translate); short
                                                 prose is the turn's header instead (header.py)

The answer's chunks join back to the answer exactly; so do a prompt's when its prose is long.
"""
import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass

from shared.memory.header import HEADER_MAX_TOKENS, prose_of
from shared.memory.segment import segment
from shared.memory.splitters import SPLITTERS
from shared.memory.tokens import count_tokens

CHUNKER_VERSION = "1"
MIN_TOKENS = 50
MAX_TOKENS = 400          # a target, not a hard limit: merging and atomic blocks can go a little over

Count = Callable[[str], int]


@dataclass(frozen=True)
class ChunkDraft:
    """A chunk before it gets an id, terms and embeddings (the memory service adds those)."""
    source: str           # prompt | response
    kind: str             # prose | code | log | math | list | table
    lang: str | None
    ord: int              # position in the turn: prompt chunks first, then the answer's
    content: str
    tokens: int
    content_hash: str


def chunk_text(text: str, count: Count = count_tokens, keep_prose: bool = True) -> list[tuple[str, str | None, str]]:
    """(kind, lang, content) pieces of one text. keep_prose=False drops the prose segments
    (a prompt whose prose is the header)."""
    pieces: list[tuple[str, str | None, str]] = []
    for seg in segment(text):
        if seg.kind == "prose" and not keep_prose:
            continue
        parts = [seg.text] if count(seg.text) <= MAX_TOKENS else SPLITTERS[seg.kind].split(seg, MAX_TOKENS, count)
        pieces.extend((seg.kind, seg.lang, p) for p in parts)
    return normalize(pieces, count)


def normalize(pieces: list[tuple[str, str | None, str]], count: Count) -> list[tuple[str, str | None, str]]:
    """Pieces under MIN_TOKENS are merged into the next one (the last into the previous). A merged
    piece takes the kind of its largest non-prose part, so "Here's the function:" ends up with
    the code it introduces, as a code chunk."""
    groups: list[list[tuple[str, str | None, str]]] = []
    pending: list[tuple[str, str | None, str]] = []
    for p in pieces:
        pending.append(p)
        if count("".join(x[2] for x in pending)) >= MIN_TOKENS:
            groups.append(pending)
            pending = []
    if pending:
        if groups:
            groups[-1].extend(pending)
        else:
            groups.append(pending)
    return [_merge(g, count) for g in groups if "".join(x[2] for x in g).strip()]


def _merge(group: list[tuple[str, str | None, str]], count: Count) -> tuple[str, str | None, str]:
    non_prose = [p for p in group if p[0] != "prose"]
    main = max(non_prose, key=lambda p: count(p[2])) if non_prose else group[0]
    return main[0], main[1], "".join(p[2] for p in group)


def chunk_turn(user_content: str, assistant_content: str, count: Count = count_tokens) -> list[ChunkDraft]:
    long_prose = count(prose_of(user_content)) > HEADER_MAX_TOKENS
    drafts = [("prompt", *p) for p in chunk_text(user_content, count, keep_prose=long_prose)]
    drafts += [("response", *p) for p in chunk_text(assistant_content, count)]
    return [
        ChunkDraft(source=source, kind=kind, lang=lang, ord=i, content=content,
                   tokens=max(1, count(content)), content_hash=content_hash(content))
        for i, (source, kind, lang, content) in enumerate(drafts)
    ]


def content_hash(content: str) -> str:
    """sha256 of the content with whitespace noise removed (trailing spaces, line endings, runs
    of blank lines), so the same code pasted twice gets the same hash (docs/rag-planv2.md 4.6)."""
    lines = [line.rstrip() for line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------ what gets embedded / indexed
def embedding_input(header: str, content: str) -> str:
    """V1 / V2: the turn's header in front of the chunk, one vector for both (4.2)."""
    return f"Question: {header}\n{content}" if header else content


def contextual_input(context: str, content: str) -> str:
    """V3+: the LLM-generated context in front of the chunk."""
    return f"{context}\n{content}"


def terms_input(header: str, content: str) -> str:
    return f"{header}\n{content}" if header else content
