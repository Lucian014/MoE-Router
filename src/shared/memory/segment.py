"""Segmentation: split a text into typed segments (docs/rag-planv2.md 4.3, step 1).

    prose | code | log | math | list | table

The segments cover the text exactly and in order: joining their `.text` gives the original back,
so every chunk is a slice of what the user or the model wrote.

Two stages, no word lists and no programming-language knowledge:

1. Markdown structure, from a CommonMark parser (markdown-it-py) -- the same reading a chat UI
   makes of the text:
       fenced block (``` or ~~~, closed or not)   code, `lang` = the fence's info string
       $$ ... $$  and  \\[ ... \\]                    math
       table (header row + |---| row)             table
       bullet / numbered list                     list
       everything else (paragraphs, headings, quotes, indented blocks, HTML): text, see 2.
2. Code and logs pasted WITHOUT a fence, recognised by the shape of their lines inside the text
   of stage 1: a run of at least MIN_UNFENCED_LINES "technical" lines becomes a code or log
   segment, the rest is prose. A line looks like
       log    a timestamp or a [TAG] first; a file:line reference (main.py:42, List.jsx:12:15,
              "main.py", line 42); an exception name (ValueError:, java.lang.NullPointerException:);
              columns aligned with runs of spaces (kubectl / pytest output)
       code   ending in { or } (or ; on a short or symbol-heavy line) -- indented (unless it ends a sentence) -- several of {}()[];=<>
              and ending like code -- a first word in capitals with no sentence end (SQL)
   Shapes, not keywords, so they hold for any programming language. Pasted code with none of
   these shapes (a few import lines) stays prose: it is still indexed, only cut less precisely.
"""
from collections.abc import Callable
from dataclasses import dataclass
from functools import lru_cache

from markdown_it import MarkdownIt
from mdit_py_plugins.dollarmath.index import math_block_dollar
from mdit_py_plugins.texmath.index import make_block_func
from mdit_py_plugins.texmath.index import rules as texmath_rules

KINDS = ("prose", "code", "log", "math", "list", "table")
MIN_UNFENCED_LINES = 3

_CODE_SYMBOLS = set("{}()[];=<>")
_CODE_ENDINGS = tuple("{};:,)([")
_BLOCK_ENDINGS = ("{", "}")
_SENTENCE_ENDINGS = (".", "?", "!", "…")
_TIMESTAMP_SEPARATORS = set("-:/.T")


@dataclass(frozen=True)
class Segment:
    kind: str
    text: str
    lang: str | None = None          # for code: the fence's info string ("python"), when given


# ------------------------------------------------------------------ the Markdown parser
def _with_map(rule: Callable) -> Callable:
    """Make a math block rule record the lines it consumed (texmath's rules don't) and leave no
    trace when the parser only asks whether it matches (silent: the plugins push tokens anyway)."""
    def wrapped(state, start, end, silent):
        pushed, line = len(state.tokens), state.line
        found = rule(state, start, end, silent)
        if silent:
            del state.tokens[pushed:]
            state.line = line
        elif found:
            state.tokens[-1].map = [start, state.line]
        return found
    return wrapped


@lru_cache(maxsize=1)
def parser() -> MarkdownIt:
    md = MarkdownIt("commonmark").enable("table")
    interrupts = {"alt": ["paragraph", "reference", "blockquote", "list"]}
    md.block.ruler.before("fence", "math_dollars", _with_map(math_block_dollar(allow_blank_lines=False)),
                          interrupts)
    for rule in texmath_rules["brackets"]["block"]:
        md.block.ruler.before("fence", f"{rule['name']}_brackets", _with_map(make_block_func(rule)), interrupts)
    return md


def lines_of(text: str) -> list[str]:
    """Lines with their endings, split exactly where the parser splits them (\\n, \\r\\n, \\r)."""
    lines, start, i = [], 0, 0
    while i < len(text):
        if text[i] in "\r\n":
            i += 2 if text.startswith("\r\n", i) else 1
            lines.append(text[start:i])
            start = i
        else:
            i += 1
    if start < len(text):
        lines.append(text[start:])
    return lines


def blocks(text: str, level: int = 0) -> list:
    """The parser's block tokens at `level` (0 = top level) that open a block, in order; each
    has .type, .info and .map = [first line, end line)."""
    return [tok for tok in parser().parse(text)
            if tok.level == level and tok.map is not None and tok.nesting >= 0]


def _kind(token_type: str) -> str:
    if token_type == "fence":
        return "code"
    if token_type.startswith("math_block"):
        return "math"
    if token_type == "table_open":
        return "table"
    if token_type in ("bullet_list_open", "ordered_list_open"):
        return "list"
    return "text"                     # judged line by line in stage 2


def segment(text: str) -> list[Segment]:
    lines = lines_of(text)
    pieces: list[tuple[str, str, str | None]] = []          # (kind, text, lang)
    covered = 0
    for tok in blocks(text):
        start, end = tok.map
        if start > covered:                                  # blank lines between blocks
            pieces.append(("text", "".join(lines[covered:start]), None))
        info = tok.info.split() if tok.type == "fence" else []
        pieces.append((_kind(tok.type), "".join(lines[start:end]), info[0].lower() if info else None))
        covered = end
    if covered < len(lines):
        pieces.append(("text", "".join(lines[covered:]), None))

    segments: list[Segment] = []
    for kind, piece, lang in _join_text(pieces):
        if kind == "text":
            segments.extend(_split_unfenced(piece))
        else:
            segments.append(Segment(kind, piece, lang))
    return _merge_neighbours(segments)


def _join_text(pieces):
    """Consecutive text pieces become one, so a run of unfenced code split by blank lines (two
    Markdown paragraphs) is judged as a whole."""
    out: list[tuple[str, str, str | None]] = []
    for p in pieces:
        if out and p[0] == "text" and out[-1][0] == "text":
            out[-1] = ("text", out[-1][1] + p[1], None)
        else:
            out.append(p)
    return out


# ------------------------------------------------------------------ unfenced code and logs
def _looks_like_log(body: str) -> bool:
    first = body.split()[0]
    bare = first.strip("[]()")
    # 2024-05-01 / 12:03:44 / [2024-05-01T12:03:44]
    if bare[:1].isdigit() and sum(c.isdigit() for c in bare) >= 4 and _TIMESTAMP_SEPARATORS & set(bare):
        return True
    # [INFO] / [WARN]: a short tag in brackets, first
    if first.startswith("[") and first.rstrip(":").endswith("]") and first.strip("[]:").isalpha():
        return True
    # columns aligned with runs of spaces: kubectl describe, pytest, tabular tool output
    if "   " in body or body.count("  ") >= 2:
        return True
    tokens = [t.strip('",') for t in body.replace("(", " ").replace(")", " ").split()]
    # ValueError: / jose.exceptions.ExpiredSignatureError: / java.lang.NullPointerException:
    if any(_is_exception_name(t) for t in tokens):
        return True
    # a file:line reference: main.py:42 / List.jsx:12:15 / at Foo.bar(Foo.java:12)
    for t in tokens:
        path, colon, position = t.partition(":")
        if colon and position.split(":")[0].isdigit() and _is_file_name(path):
            return True
    # File "main.py", line 42
    quoted_file = '"' in body and any(_is_file_name(t) for t in tokens)
    return quoted_file and any(t.isdigit() for t in tokens)


def _is_exception_name(token: str) -> bool:
    """`Name:` where Name is an identifier (or dotted path) in CamelCase with at least two
    capitals in its last part: ValueError:, NullPointerException:, jose.exceptions.JWTError:."""
    if not token.endswith(":") or len(token) < 4:
        return False
    parts = token[:-1].split(".")
    last = parts[-1]
    return (all(p.isidentifier() for p in parts) and any(c.islower() for c in last)
            and sum(c.isupper() for c in last) >= 2 and last[0].isupper())


def _is_file_name(path: str) -> bool:
    stem, dot, ext = path.rpartition(".")
    return bool(dot and stem) and ext.isalnum() and not ext.isdigit() and len(ext) <= 5


def _line_kind(line: str) -> str | None:
    """'log', 'code' or None (prose) for one non-blank line, from its shape only."""
    text = line.rstrip("\r\n")
    body = text.strip()
    if not body:
        return None
    if _looks_like_log(body):
        return "log"
    symbols = sum(c in _CODE_SYMBOLS for c in body)
    if body.endswith(_BLOCK_ENDINGS):
        return "code"
    # `;` ends statements, but also items of an enumeration in prose: code if short or symbolic
    if body.endswith(";") and (len(body.split()) <= 3 or symbols >= 2):
        return "code"
    indent = len(text) - len(text.lstrip(" \t"))
    if (indent >= 2 or text.startswith("\t")) and not body.endswith(_SENTENCE_ENDINGS):
        return "code"
    if symbols >= 2 and (symbols / len(body) >= 0.12 or body.endswith(_CODE_ENDINGS)):
        return "code"
    first = body.split()[0]
    if len(first) >= 3 and first.isalpha() and first.isupper() and not body.endswith(_SENTENCE_ENDINGS):
        return "code"
    return None


def _split_unfenced(text: str) -> list[Segment]:
    """Prose with runs of >= MIN_UNFENCED_LINES code / log lines cut out as code / log segments.
    Blank lines inside a run don't end it; the run's kind is the majority of its lines."""
    lines = lines_of(text)
    kinds = [_line_kind(line) if line.strip() else "blank" for line in lines]
    out: list[Segment] = []
    start = 0                         # start of the pending prose
    i = 0
    while i < len(lines):
        if kinds[i] not in ("code", "log"):
            i += 1
            continue
        j, counted = i, []
        while j < len(lines) and kinds[j] in ("code", "log", "blank"):
            if kinds[j] != "blank":
                counted.append(kinds[j])
            j += 1
        while j > i and kinds[j - 1] == "blank":            # trailing blanks go back to prose
            j -= 1
        if len(counted) >= MIN_UNFENCED_LINES:
            if start < i:
                out.append(Segment("prose", "".join(lines[start:i])))
            kind = "log" if counted.count("log") * 2 >= len(counted) else "code"
            out.append(Segment(kind, "".join(lines[i:j])))
            start = j
        i = max(j, i + 1)
    if start < len(lines):
        out.append(Segment("prose", "".join(lines[start:])))
    return out


def _merge_neighbours(segments: list[Segment]) -> list[Segment]:
    """Two prose segments in a row (e.g. blank lines between blocks) become one."""
    out: list[Segment] = []
    for s in segments:
        if out and s.kind == "prose" and out[-1].kind == "prose":
            out[-1] = Segment("prose", out[-1].text + s.text)
        else:
            out.append(s)
    return out
