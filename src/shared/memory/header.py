"""The turn header and the dense query (docs/rag-planv2.md 4.2, 4.4).

Header: what a turn was about, put in front of each of its chunks before embedding / BM25 and
shown as the title of its excerpts in the context. From the prompt's prose:
    prose <= HEADER_MAX_TOKENS     the whole prose (most prompts pass unchanged)
    longer                         sentences with "?" first, then from the end backwards, up to
                                   HEADER_MAX_TOKENS, kept in their original order
    no prose (only code / a log)   the first sentence of the answer (from_answer=True)

Dense query: the same selection on the current prompt, so a pasted traceback next to a question
doesn't dilute the vector; a prompt with no prose (only an error, a log, code) is used whole, or
its LAST max_tokens tokens -- in a traceback the exception and its frame are at the end. The
truncation is done here: Ollama would cut the end instead.
"""
import re
from collections.abc import Callable

from shared.memory.segment import segment
from shared.memory.tokens import count_tokens

HEADER_MAX_TOKENS = 150
DENSE_QUERY_MAX_TOKENS = 512

Count = Callable[[str], int]

_SENTENCE = re.compile(r"[^.!?…\n]+(?:[.!?…]+|\n|$)")


def prose_of(text: str) -> str:
    """The prompt without its code, logs, formulas and tables, whitespace collapsed."""
    prose = " ".join(s.text for s in segment(text) if s.kind in ("prose", "list"))
    return " ".join(prose.split())


def sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE.findall(text) if s.strip()]


def select_sentences(prose: str, max_tokens: int, count: Count = count_tokens) -> str:
    if count(prose) <= max_tokens:
        return prose
    sents = sentences(prose)
    order = [i for i, s in enumerate(sents) if "?" in s]
    order += [i for i in reversed(range(len(sents))) if i not in order]
    chosen: set[int] = set()
    for i in order:
        candidate = " ".join(sents[j] for j in sorted(chosen | {i}))
        if count(candidate) <= max_tokens:
            chosen.add(i)
    if not chosen and sents:                 # one huge sentence: its end, where questions are
        return tail_tokens(sents[order[0]], max_tokens, count)
    return " ".join(sents[j] for j in sorted(chosen))


def extract_question(user_content: str, assistant_content: str,
                     count: Count = count_tokens) -> tuple[str, bool]:
    """(header, from_answer)."""
    prose = prose_of(user_content)
    if prose:
        return select_sentences(prose, HEADER_MAX_TOKENS, count), False
    answer = sentences(prose_of(assistant_content))
    return (select_sentences(answer[0], HEADER_MAX_TOKENS, count) if answer else ""), True


def build_dense_query(prompt: str, max_tokens: int = DENSE_QUERY_MAX_TOKENS,
                      count: Count = count_tokens) -> str:
    prose = prose_of(prompt)
    if prose:
        return select_sentences(prose, HEADER_MAX_TOKENS, count)
    return tail_tokens(prompt.strip(), max_tokens, count)


def tail_tokens(text: str, max_tokens: int, count: Count = count_tokens) -> str:
    """The longest suffix of `text` within max_tokens, starting at a line or word boundary."""
    if count(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:                           # smallest start index that fits
        mid = (lo + hi) // 2
        if count(text[mid:]) <= max_tokens:
            hi = mid
        else:
            lo = mid + 1
    newline = text.find("\n", lo)
    if 0 <= newline < lo + 200:
        return text[newline + 1:]
    space = text.find(" ", lo)
    return text[space + 1:] if 0 <= space < lo + 50 else text[lo:]
