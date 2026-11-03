"""
router/classifier_input.py -- what the classifiers are given when a prompt is longer than they read.

A Laya classifier reads max_len tokens (512), of which up to head_max_len (256) are its question and
options: the prompt gets the rest, ~250 tokens, and Laya keeps the START of a longer prompt without a
warning. A pasted document with the request written after it ("<article>\n\nRezumă textul de mai
sus.") is then classified on the article alone (scripts/classifier_eval/long_prompts.py: 42% correct
on prose documents with the request at the end).

    edges   (CLASSIFIER_INPUT=edges, the default)
            a prompt that fits is given as it is (most prompts: the distribution the classifiers were
            trained on); a longer one becomes its short first and / or last paragraph -- where a
            request next to a pasted document usually is -- followed by the start of the rest, which
            says what the document is (an article, code, a log). No short paragraph at either end:
            the start and the end of the prompt, "[…]" between (head_tail)
    raw     the prompt as it is (Laya cuts it)

Shapes only (blank lines, paragraph lengths), no word lists: the rule holds in any language.
Measured with the domain classifier: 81% -> 97% on long prompts, 42% -> 94% on prose documents with
the request at the end, no loss with the request at the start.
"""
from collections.abc import Callable

EDGE_MAX_TOKENS = 60                   # a paragraph this short at either end is taken as the request
GAP = "\n[…]\n"
STRATEGIES = ("edges", "raw")

Count = Callable[[str], int]


def paragraphs(text: str) -> list[str]:
    """Blocks of text separated by blank lines."""
    out, current = [], []
    for line in text.splitlines():
        if line.strip():
            current.append(line)
        elif current:
            out.append("\n".join(current))
            current = []
    if current:
        out.append("\n".join(current))
    return out


def head(text: str, budget: int, count: Count) -> str:
    """The longest prefix within `budget` tokens (cut on characters, so no re-decoded token text)."""
    if count(text) <= budget:
        return text
    lo, hi = 0, len(text)
    while lo < hi:                           # largest end index that fits
        mid = (lo + hi + 1) // 2
        if count(text[:mid]) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo]


def tail(text: str, budget: int, count: Count) -> str:
    """The longest suffix within `budget` tokens."""
    if count(text) <= budget:
        return text
    lo, hi = 0, len(text)
    while lo < hi:                           # smallest start index that fits
        mid = (lo + hi) // 2
        if count(text[mid:]) <= budget:
            hi = mid
        else:
            lo = mid + 1
    return text[lo:]


def head_tail(text: str, budget: int, count: Count) -> str:
    if count(text) <= budget:
        return text
    half = max(1, (budget - count(GAP)) // 2)
    return head(text, half, count) + GAP + tail(text, half, count)


def edges(text: str, budget: int, count: Count, edge_max: int = EDGE_MAX_TOKENS) -> str:
    """A short first and / or last paragraph (the request, usually) + the start of what is between.
    No short paragraph at either end: head_tail."""
    if count(text) <= budget:
        return text
    paras = paragraphs(text)
    if len(paras) < 2:
        return head_tail(text, budget, count)
    request, body = [], paras
    if count(paras[0]) <= edge_max:
        request.append(paras[0])
        body = body[1:]
    if count(paras[-1]) <= edge_max and len(body) > 1:
        request.append(paras[-1])
        body = body[:-1]
    if not request:
        return head_tail(text, budget, count)
    asked = "\n\n".join(request)
    room = budget - count(asked + "\n\n")
    if room <= 0:
        return asked
    return asked + "\n\n" + head("\n\n".join(body), room, count)


def classifier_text(prompt: str, budget: int, count: Count, strategy: str = "edges") -> tuple[str, str]:
    """(the text to classify, how it was made): "full" -- the prompt fits; "edges" -- shaped here;
    "cut" -- too long and given raw, so Laya keeps its start."""
    if count(prompt) <= budget:
        return prompt, "full"
    if strategy == "raw":
        return prompt, "cut"
    return edges(prompt, budget, count), "edges"
