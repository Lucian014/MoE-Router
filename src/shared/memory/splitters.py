"""Splitters: cut one segment that is too long into pieces of at most ~max_tokens
(docs/rag-planv2.md 4.3, step 2). One strategy per segment kind, never per conversation category.

Every splitter returns pieces that join back to the segment's text exactly. Small parts are packed
greedily up to max_tokens, so a long segment becomes a few full pieces, not many tiny ones.
"""
import ast
import re
from collections.abc import Callable
from typing import Protocol

from shared.memory.segment import _LIST_ITEM as LIST_ITEM
from shared.memory.segment import Segment

Count = Callable[[str], int]

_HEADING = re.compile(r"^#{1,6}\s")
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")
_PYTHON = {"python", "py", "python3"}


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
    return text.splitlines(keepends=True)


def _split_before(lines: list[str], is_boundary) -> list[str]:
    """Group lines into blocks, starting a new block at each boundary line."""
    blocks: list[str] = []
    for line in lines:
        if not blocks or is_boundary(line):
            blocks.append(line)
        else:
            blocks[-1] += line
    return blocks


def _paragraphs(text: str) -> list[str]:
    """Paragraphs with their trailing blank lines."""
    parts = re.split(r"(\n[ \t]*\n)", text)
    out = []
    for i in range(0, len(parts), 2):
        out.append(parts[i] + (parts[i + 1] if i + 1 < len(parts) else ""))
    return [p for p in out if p]


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
    """Headings, then paragraphs, then sentences, then a hard cut."""

    def split(self, seg: Segment, max_tokens: int, count: Count) -> list[str]:
        parts = _split_before(_split_lines(seg.text), lambda line: bool(_HEADING.match(line)))
        parts = _refine(parts, max_tokens, count, _paragraphs)
        parts = _refine(parts, max_tokens, count, _sentences)
        parts = _refine(parts, max_tokens, count, lambda p: hard_cut(p, max_tokens, count))
        return pack(parts, max_tokens, count)


class CodeSplitter:
    """Python: top-level definitions (ast), decorators and preceding comments with their
    definition, classes over the limit by method. Other languages, or code that doesn't parse:
    blank lines, then lines, then a hard cut. Fence lines stay with the first / last piece."""

    def split(self, seg: Segment, max_tokens: int, count: Count) -> list[str]:
        lines = _split_lines(seg.text)
        head, body, tail = _strip_fence(lines)
        units = None
        if seg.lang in _PYTHON or seg.lang is None:
            units = _python_units(body, max_tokens, count)
        if units is None:
            units = _paragraphs("".join(body))
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
        items = _split_before(_split_lines(seg.text), lambda line: bool(LIST_ITEM.match(line)))
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
def _strip_fence(lines: list[str]) -> tuple[list[str], list[str], list[str]]:
    if lines and re.match(r"^ {0,3}(`{3,}|~{3,})", lines[0]):
        head, rest = lines[:1], lines[1:]
        if rest and re.match(r"^ {0,3}(`{3,}|~{3,})\s*$", rest[-1]):
            return head, rest[:-1], rest[-1:]
        return head, rest, []
    return [], lines, []


def _python_units(lines: list[str], max_tokens: int, count: Count) -> list[str] | None:
    """Top-level statements as units: a unit starts at a definition's first decorator or at the
    comments / blank lines right after the previous unit. None if the code doesn't parse."""
    source = "".join(lines)
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return None
    if not tree.body:
        return None
    units = []
    for node, a, b in zip(tree.body, *_bounds(tree.body, 0, len(lines))):
        text = "".join(lines[a:b])
        if isinstance(node, ast.ClassDef) and count(text) > max_tokens and len(node.body) > 1:
            # the class line (+ docstring / first statement) is the first unit, then one per method
            for _, x, y in zip(node.body, *_bounds(node.body, a, b)):
                units.append("".join(lines[x:y]))
        else:
            units.append(text)
    return units


def _bounds(nodes: list[ast.stmt], start: int, end: int) -> tuple[list[int], list[int]]:
    """Line ranges [a, b) per statement: each runs from the end of the previous one, so the
    comments, blank lines and decorators before a definition go with it."""
    ends = [n.end_lineno for n in nodes[:-1]]
    return [start] + ends, ends + [end]
