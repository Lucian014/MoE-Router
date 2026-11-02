"""Segmentation: split a text into typed segments (docs/rag-planv2.md 4.3, step 1).

    prose | code | log | math | list | table

The segments cover the text exactly and in order: joining their `.text` gives the original back,
so every chunk is a slice of what the user or the model wrote. Recognised by structure only
(no word lists), the same way for prompts and answers:

    code   fenced blocks (``` or ~~~; closed by the same character, at least as many; an unclosed
           fence, e.g. an answer cut at max_tokens, runs to the end); unfenced code: runs of at
           least 3 code-looking lines (indentation, symbol density, `def` / `import` / `{`...)
    log    unfenced runs of at least 3 log-looking lines (Traceback, `File "...", line N`,
           `SomethingError:`, timestamps, `at pkg.Class(File.java:12)`)
    math   $$ ... $$ and \\[ ... \\] blocks
    table  markdown tables (header row + |---| separator)
    list   consecutive list items (-, *, +, 1.), with their indented continuation lines
    prose  everything else; markdown headings stay in prose as section boundaries
"""
import re
from dataclasses import dataclass

KINDS = ("prose", "code", "log", "math", "list", "table")

_FENCE_OPEN = re.compile(r"^ {0,3}(`{3,}|~{3,})\s*([^\s`]*)")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_LIST_ITEM = re.compile(r"^\s{0,3}([-*+]|\d{1,3}[.)])\s+\S")

_LOG_LINE = re.compile(
    r"^\s*(Traceback \(most recent call last\)"
    r"|File \".+\", line \d+"
    r"|[\w.]*(Error|Exception|Warning)\b:?"
    r"|at [\w.$<>]+\(.*:\d+\)"
    r"|\[?\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}"
    r"|\[?(DEBUG|INFO|WARN|WARNING|ERROR|CRITICAL|FATAL)\]?[\s:])"
)
_CODE_START = re.compile(
    r"^\s*(def |class |import |from \S+ import |return\b|if .*:$|elif .*:$|else:|for .*:$|while .*:$|"
    r"try:|except\b|with .*:$|@\w+|async def |function\b|const |let |var |public |private |protected |"
    r"#include|package |using |fn |func |SELECT\b|INSERT\b|UPDATE\b|CREATE\b|<\w+[^>]*>|</\w+>|\}|\{)"
)
_CODE_END = re.compile(r"[{};:]\s*$|\)\s*$")
_SYMBOLS = set("{}()[];=<>")
MIN_UNFENCED_LINES = 3


@dataclass(frozen=True)
class Segment:
    kind: str
    text: str
    lang: str | None = None          # for code: the fence's info string ("python"), when given


def segment(text: str) -> list[Segment]:
    lines = text.splitlines(keepends=True)
    segments: list[Segment] = []
    prose: list[str] = []

    def flush_prose():
        if prose:
            segments.extend(_split_unfenced("".join(prose)))
            prose.clear()

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        fence = _FENCE_OPEN.match(line)
        if fence:
            end = _fence_end(lines, i, fence.group(1))
            flush_prose()
            lang = fence.group(2).lower() or None
            segments.append(Segment("code", "".join(lines[i:end]), lang))
            i = end
            continue

        if stripped.startswith("$$") or stripped.startswith("\\["):
            end = _math_end(lines, i)
            flush_prose()
            segments.append(Segment("math", "".join(lines[i:end])))
            i = end
            continue

        if stripped.startswith("|") and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            end = i + 2
            while end < len(lines) and lines[end].strip().startswith("|"):
                end += 1
            flush_prose()
            segments.append(Segment("table", "".join(lines[i:end])))
            i = end
            continue

        if _LIST_ITEM.match(line):
            end = _list_end(lines, i)
            flush_prose()
            segments.append(Segment("list", "".join(lines[i:end])))
            i = end
            continue

        prose.append(line)
        i += 1

    flush_prose()
    return _merge_neighbours(segments)


# ------------------------------------------------------------------ block ends
def _fence_end(lines: list[str], start: int, opener: str) -> int:
    """Index after the closing fence line; len(lines) for an unclosed fence."""
    char, size = opener[0], len(opener)
    closing = re.compile(rf"^ {{0,3}}{re.escape(char)}{{{size},}}\s*$")
    for j in range(start + 1, len(lines)):
        if closing.match(lines[j]):
            return j + 1
    return len(lines)


def _math_end(lines: list[str], start: int) -> int:
    first = lines[start].strip()
    opener, closer = ("$$", "$$") if first.startswith("$$") else ("\\[", "\\]")
    if closer in first[len(opener):]:                    # $$ x $$ on one line
        return start + 1
    for j in range(start + 1, len(lines)):
        if closer in lines[j]:
            return j + 1
    return len(lines)


def _list_end(lines: list[str], start: int) -> int:
    """Items plus indented continuation lines; a blank line ends the list unless the next
    non-blank line is another item or indented continuation."""
    j = start + 1
    while j < len(lines):
        line = lines[j]
        if _LIST_ITEM.match(line) or (line.strip() and line[:1] in " \t"):
            j += 1
            continue
        if not line.strip():
            k = j
            while k < len(lines) and not lines[k].strip():
                k += 1
            if k < len(lines) and (_LIST_ITEM.match(lines[k]) or lines[k][:1] in " \t"):
                j = k
                continue
        break
    return j


# ------------------------------------------------------------------ unfenced code and logs
def _line_kind(line: str) -> str | None:
    """'log', 'code' or None (prose) for one non-blank line."""
    if _LOG_LINE.match(line):
        return "log"
    s = line.rstrip("\n")
    if not s.strip():
        return None
    if s[:1] in " \t" and len(s) - len(s.lstrip()) >= 4:
        return "code"
    if _CODE_START.match(s):
        return "code"
    body = s.strip()
    symbols = sum(c in _SYMBOLS for c in body)
    if symbols >= 2 and (symbols / len(body) >= 0.12 or _CODE_END.search(body)):
        return "code"
    return None


def _split_unfenced(text: str) -> list[Segment]:
    """Prose with runs of >= MIN_UNFENCED_LINES code / log lines cut out as code / log segments.
    Blank lines inside a run don't end it."""
    lines = text.splitlines(keepends=True)
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
    """Two prose segments in a row (e.g. around a filtered-out block) become one."""
    out: list[Segment] = []
    for s in segments:
        if out and s.kind == "prose" and out[-1].kind == "prose":
            out[-1] = Segment("prose", out[-1].text + s.text)
        else:
            out.append(s)
    return out
