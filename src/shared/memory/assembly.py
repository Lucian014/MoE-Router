"""Context assembly: fit a session's memory into the chosen model's token budget
(docs/rag-planv2.md 4.7). Pure functions, no I/O: the router loads the turns (R1) and the
fragments (R2/R3) and passes them in.

    available = context_window - max_output_tokens - system - prompt - margin

1. the last turn is always included (its texts are cut with "[…]" if it would take more than
   last_turn_max_share of `available`)
2. older turns, newest first and whole, while they fit in recent_max_share of `available`
   (all of `available` when there are no fragments); the first that doesn't fit stops the walk,
   so the included turns are always the most recent, contiguous ones
3. fragments from turns that are not included whole, in the given (relevance) order, while they fit
4. space still left: older turns whole again; a turn added this way drops its fragments

Messages, in chronological order:
    system   instructions
    system   relevant excerpts, grouped by turn ("[turn N] Question: ...")   (only with fragments)
    user / assistant pairs of the included turns
    user     the current prompt
"""
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from shared.memory.tokens import MESSAGE_OVERHEAD_TOKENS, count_tokens

ELLIPSIS = " […]"
EXCERPTS_TITLE = "Relevant excerpts from earlier in this conversation:"


@dataclass(frozen=True)
class TurnText:
    turn: int
    user: str
    assistant: str


@dataclass(frozen=True)
class Fragment:
    chunk_id: int
    turn: int
    ord: int
    content: str
    header: str = ""                  # the turn's header (docs/rag-planv2.md 4.2)
    header_from_answer: bool = False  # header taken from the answer: shown as "Topic", not "Question"


@dataclass(frozen=True)
class Budget:
    context_window: int
    max_output_tokens: int
    margin: float = 0.10              # the pool's tokenizers differ from the reference one
    recent_max_share: float = 0.6
    last_turn_max_share: float = 0.5


@dataclass
class AssembledContext:
    messages: list[dict[str, str]]
    recent_turns: int                 # turns included whole
    fragment_ids: list[int] = field(default_factory=list)
    input_tokens: int = 0             # estimated, with the reference tokenizer
    last_turn_truncated: bool = False


class PromptTooLongError(ValueError):
    """The prompt alone doesn't fit the model's context."""


def truncate_to(text: str, max_tokens: int, count: Callable[[str], int] = count_tokens) -> str:
    """The longest prefix of `text` (plus " […]") that counts at most `max_tokens`."""
    if count(text) <= max_tokens:
        return text
    lo, hi = 0, len(text)
    while lo < hi:                    # largest prefix length that still fits
        mid = (lo + hi + 1) // 2
        if count(text[:mid] + ELLIPSIS) <= max_tokens:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + ELLIPSIS if lo else ""


def build_messages(
    system: str,
    turns: Sequence[TurnText],
    prompt: str,
    budget: Budget,
    fragments: Sequence[Fragment] = (),
    count: Callable[[str], int] = count_tokens,
) -> AssembledContext:
    """`turns` oldest first (as R1 returns them); `fragments` in relevance order."""

    def cost(text: str) -> int:
        return count(text) + MESSAGE_OVERHEAD_TOKENS

    def turn_cost(t: TurnText) -> int:
        return cost(t.user) + cost(t.assistant)

    fixed = cost(system) + cost(prompt)
    available = (budget.context_window - budget.max_output_tokens - fixed
                 - int(budget.margin * budget.context_window))
    if available < 0:
        raise PromptTooLongError(
            f"the prompt needs {fixed} tokens with the system prompt, the model has "
            f"{budget.context_window} (of which {budget.max_output_tokens} for the answer)")

    # 1. the last turn, cut if it's too long
    included: list[TurnText] = []
    last_truncated = False
    if turns:
        last = turns[-1]
        cap = int(budget.last_turn_max_share * available)
        if turn_cost(last) > cap:
            last, last_truncated = _shrink_turn(last, cap, cost, count), True
        if turn_cost(last) <= available:
            included.append(last)
    used = sum(turn_cost(t) for t in included)

    # 2. older turns, newest first, whole, contiguous
    recent_cap = int(budget.recent_max_share * available) if fragments else available
    older = list(turns[:-1])
    while older and used + turn_cost(older[-1]) <= recent_cap:
        t = older.pop()
        included.insert(0, t)
        used += turn_cost(t)

    # 3. fragments of the turns that are not included whole
    whole = {t.turn for t in included}
    chosen = _fit_fragments([f for f in fragments if f.turn not in whole], available - used, count)

    # 4. space left: more older turns whole, dropping their fragments
    while older:
        t = older[-1]
        freed = [f for f in chosen if f.turn == t.turn]
        rest = [f for f in chosen if f.turn != t.turn]
        if used + turn_cost(t) + _excerpts_cost(rest, count) > available:
            break
        older.pop()
        included.insert(0, t)
        used += turn_cost(t)
        if freed:
            chosen = rest

    messages = [{"role": "system", "content": system}]
    excerpts = render_excerpts(chosen)
    if excerpts:
        messages.append({"role": "system", "content": excerpts})
    for t in included:
        messages += [{"role": "user", "content": t.user}, {"role": "assistant", "content": t.assistant}]
    messages.append({"role": "user", "content": prompt})

    return AssembledContext(
        messages=messages,
        recent_turns=len(included),
        fragment_ids=[f.chunk_id for f in chosen],
        input_tokens=fixed + used + _excerpts_cost(chosen, count),
        last_turn_truncated=last_truncated,
    )


def _shrink_turn(t: TurnText, cap: int, cost, count) -> TurnText:
    """Cut the answer first (the prompt is usually the short part); if the prompt alone is too
    long, both get half."""
    room = cap - cost(t.user) - MESSAGE_OVERHEAD_TOKENS
    if room >= cap // 4:
        return TurnText(t.turn, t.user, truncate_to(t.assistant, room, count))
    half = cap // 2 - MESSAGE_OVERHEAD_TOKENS
    return TurnText(t.turn, truncate_to(t.user, half, count), truncate_to(t.assistant, half, count))


def _fit_fragments(candidates: Sequence[Fragment], room: int, count) -> list[Fragment]:
    chosen: list[Fragment] = []
    for f in candidates:
        if _excerpts_cost(chosen + [f], count) <= room:
            chosen.append(f)
    return chosen


def _excerpts_cost(fragments: Sequence[Fragment], count) -> int:
    text = render_excerpts(fragments)
    return count(text) + MESSAGE_OVERHEAD_TOKENS if text else 0


def render_excerpts(fragments: Sequence[Fragment]) -> str:
    """Fragments grouped by turn, turns and chunks in conversation order, "[…]" between chunks
    that are not consecutive in their turn."""
    if not fragments:
        return ""
    by_turn: dict[int, list[Fragment]] = {}
    for f in fragments:
        by_turn.setdefault(f.turn, []).append(f)
    parts = [EXCERPTS_TITLE]
    for turn in sorted(by_turn):
        group = sorted(by_turn[turn], key=lambda f: f.ord)
        head = group[0]
        label = "Topic" if head.header_from_answer else "Question"
        lines = [f"[turn {turn}] {label}: {head.header}" if head.header else f"[turn {turn}]"]
        prev = None
        for f in group:
            if prev is not None and f.ord != prev + 1:
                lines.append("[…]")
            lines.append(f.content)
            prev = f.ord
        parts.append("\n".join(lines))
    return "\n\n".join(parts)
