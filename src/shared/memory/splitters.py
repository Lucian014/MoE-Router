"""Splitters: cut one segment that is too long into pieces of at most ~max_tokens
(docs/rag-planv2.md 4.3, step 2). One strategy per segment kind, never per conversation category.

Every splitter returns pieces that join back to the segment's text exactly. Small parts are packed
greedily up to max_tokens, so a long segment becomes a few full pieces, not many tiny ones.

Where Markdown structure matters (headings, list items) the boundaries come from the same parser
as the segmentation; code is cut by indentation and blank lines, the same way in any language.
"""
import re
from collections.abc import Callable
from typing import Protocol

from shared.memory.segment import Segment, blocks, lines_of

Count = Callable[[str], int]

_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_FENCE_CHARS = ("`", "~")


class Splitter(Protocol):
    def split(self, seg: Segment, max_tokens: int, count: Count) -> list[str]: ...


# ------------------------------------------------------------------ helpers
def pack(parts: list[str], max_tokens: int, count: Count) -> list[str]:
    """Join consecutive parts while the result stays within max_tokens."""
    out: list[str] = []
    for p in parts:
        if out and count(out[-1] + p) <= max_tokens:
            out[-1] += p
        else:
            out.append(p)
    return out


def hard_cut(text: str, max_tokens: int, count: Count) -> list[str]:
    """Last resort (a minified JSON line, a sentence of 2000 words): prefixes of at most
    max_tokens, preferring to cut after whitespace."""
    out = []
    while text and count(text) > max_tokens:
        lo, hi = 1, len(text)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if count(text[:mid]) <= max_tokens:
                lo = mid
            else:
                hi = mid - 1
        space = text.rfind(" ", 0, lo)
        cut = space + 1 if space > lo // 2 else lo
        out.append(text[:cut])
        text = text[cut:]
    if text:
        out.append(text)
    return out


def _split_lines(text: str) -> list[str]:
    return lines_of(text)


def _split_at(lines: list[str], starts: set[int]) -> list[str]:
    """Group lines into blocks, a new block starting at each line index in `starts`."""
    out: list[str] = []
    for i, line in enumerate(lines):
        if not out or i in starts:
            out.append(line)
        else:
            out[-1] += line
    return out


def _paragraphs(text: str) -> list[str]:
    """Paragraphs with their trailing blank lines: a new one starts at a non-blank line that
    follows a blank one."""
    lines = lines_of(text)
    starts = {i for i in range(1, len(lines)) if lines[i].strip() and not lines[i - 1].strip()}
    return _split_at(lines, starts)


def _sentences(text: str) -> list[str]:
    cuts = [m.end() for m in _SENTENCE_END.finditer(text)]
    starts = [0] + cuts
    return [text[a:b] for a, b in zip(starts, cuts + [len(text)]) if text[a:b]]


def _refine(parts: list[str], max_tokens: int, count: Count, finer: Callable[[str], list[str]]) -> list[str]:
    """Split only the parts that are still too long, with the next finer strategy."""
    out = []
    for p in parts:
        out.extend(finer(p) if count(p) > max_tokens else [p])
    return out


# ------------------------------------------------------------------ splitters
class ProseSplitter:
    """Headings (from the Markdown parser), then paragraphs, then sentences, then a hard cut."""

    def split(self, seg: Segment, max_tokens: int, count: Count) -> list[str]:
        headings = {tok.map[0] for tok in blocks(seg.text) if tok.type == "heading_open"}
        parts = _split_at(_split_lines(seg.text), headings)
        parts = _refine(parts, max_tokens, count, _paragraphs)
        parts = _refine(parts, max_tokens, count, _sentences)
        parts = _refine(parts, max_tokens, count, lambda p: hard_cut(p, max_tokens, count))
        return pack(parts, max_tokens, count)


class CodeSplitter:
    """By indentation, in any language: a unit starts at an unindented line after a blank line
    (a function, a class, a block of statements), so decorators and comments right above a
    definition stay with it. A unit still too long is cut the same way one indentation level
    deeper (the methods of a class), then by blank lines, lines, a hard cut. Fence lines stay with
    the first / last piece."""

    def split(self, seg: Segment, max_tokens: int, count: Count) -> list[str]:
        head, body, tail = _strip_fence(_split_lines(seg.text))
        units = []
        for unit in _units_at_level(body, 0):
            if count("".join(unit)) > max_tokens and len(unit) > 1:
                inner = _units_at_level(unit, _inner_indent(unit))
                units.extend("".join(u) for u in inner)
            else:
                units.append("".join(unit))
        units = _refine(units, max_tokens, count, _paragraphs)
        units = _refine(units, max_tokens, count, _split_lines)
        units = _refine(units, max_tokens, count, lambda p: hard_cut(p, max_tokens, count))
        pieces = pack(units, max_tokens, count) or [""]
        pieces[0] = "".join(head) + pieces[0]
        pieces[-1] += "".join(tail)
        return [p for p in pieces if p]


class LogSplitter:
    """By lines, packed from the end: the exception and the last frames stay together."""

    def split(self, seg: Segment, max_tokens: int, count: Count) -> list[str]:
        lines = _refine(_split_lines(seg.text), max_tokens, count, lambda p: hard_cut(p, max_tokens, count))
        pieces, current = [], ""
        for line in reversed(lines):
            if current and count(line + current) > max_tokens:
                pieces.append(current)
                current = line
            else:
                current = line + current
        if current:
            pieces.append(current)
        return pieces[::-1]


class ListSplitter:
    """Between items, never inside one."""

    def split(self, seg: Segment, max_tokens: int, count: Count) -> list[str]:
        items_start = {tok.map[0] for tok in blocks(seg.text, level=1) if tok.type == "list_item_open"}
        items = _split_at(_split_lines(seg.text), items_start)
        items = _refine(items, max_tokens, count, lambda p: ProseSplitter().split(Segment("prose", p), max_tokens, count))
        return pack(items, max_tokens, count)


class AtomicSplitter:
    """Formulas and tables are never cut, even a little over the limit."""

    def split(self, seg: Segment, max_tokens: int, count: Count) -> list[str]:
        return [seg.text]


SPLITTERS: dict[str, Splitter] = {
    "prose": ProseSplitter(),
    "code": CodeSplitter(),
    "log": LogSplitter(),
    "list": ListSplitter(),
    "math": AtomicSplitter(),
    "table": AtomicSplitter(),
}


# ------------------------------------------------------------------ code helpers
def _is_fence_line(line: str, closing: bool = False) -> bool:
    body = line.strip()
    if closing:
        return len(body) >= 3 and body[0] in _FENCE_CHARS and body == body[0] * len(body)
    return line.startswith(("```", "~~~", " ```", " ~~~", "  ```", "  ~~~", "   ```", "   ~~~"))


def _strip_fence(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    """(opening fence line, the code, closing fence line); empty lists for unfenced code."""
    if lines and _is_fence_line(lines[0]):
        head, rest = lines[:1], lines[1:]
        if rest and _is_fence_line(rest[-1], closing=True):
            return head, rest[:-1], rest[-1:]
        return head, rest, []
    return [], lines, []


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _inner_indent(lines: list[str]) -> int:
    """The smallest indentation of the unit's body (its lines after the first)."""
    indents = [_indent(line) for line in lines[1:] if line.strip()]
    return min(indents) if indents else 0


def _units_at_level(lines: list[str], level: int) -> list[list[str]]:
    """Cut before every line indented exactly `level` that follows a blank line."""
    units: list[list[str]] = []
    for i, line in enumerate(lines):
        starts = (i > 0 and line.strip() and _indent(line) == level and not lines[i - 1].strip())
        if not units or starts:
            units.append([line])
        else:
            units[-1].append(line)
    return units
