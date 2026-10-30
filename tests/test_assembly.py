"""Context assembly (src/shared/memory/assembly.py). Tokens are counted as words here, so the
budgets in the tests are easy to follow; MESSAGE_OVERHEAD_TOKENS still applies per message."""
import pytest

from shared.memory.assembly import (
    Budget,
    Fragment,
    PromptTooLongError,
    TurnText,
    build_messages,
    render_excerpts,
    truncate_to,
)
from shared.memory.tokens import MESSAGE_OVERHEAD_TOKENS as O


def words(text):
    return len(text.split())


def turn(n, size=10):
    """Turn n with a 1-word question and a (size - 1)-word answer: costs size + 2 * O."""
    return TurnText(n, f"q{n}", " ".join([f"a{n}"] * (size - 1)))


def build(turns, budget, prompt="p", fragments=()):
    return build_messages("s", turns, prompt, budget, fragments, count=words)


def available(budget, prompt="p"):
    return (budget.context_window - budget.max_output_tokens - (1 + O) - (words(prompt) + O)
            - int(budget.margin * budget.context_window))


def roles(ctx):
    return [m["role"] for m in ctx.messages]


def test_no_history_is_system_plus_prompt():
    ctx = build([], Budget(1000, 100))
    assert ctx.messages == [{"role": "system", "content": "s"}, {"role": "user", "content": "p"}]
    assert ctx.recent_turns == 0


def test_everything_fits_on_a_big_model():
    turns = [turn(i) for i in range(1, 13)]
    ctx = build(turns, Budget(32768, 2048))
    assert ctx.recent_turns == 12
    assert roles(ctx) == ["system"] + ["user", "assistant"] * 12 + ["user"]
    assert ctx.messages[1]["content"] == "q1"                    # chronological


def test_small_model_keeps_the_most_recent_contiguous_turns():
    turns = [turn(i, size=100) for i in range(1, 13)]           # 108 tokens each with overheads
    budget = Budget(600, 100)                                   # available = 600-100-5-5-60 = 430
    ctx = build(turns, budget)
    assert ctx.recent_turns == 3                                # 3 * 108 = 324 <= 430 < 432
    assert [m["content"] for m in ctx.messages if m["role"] == "user"][:3] == ["q10", "q11", "q12"]


def test_last_turn_is_kept_but_cut_when_too_long():
    turns = [turn(1), TurnText(2, "q2", " ".join(["x"] * 1000))]
    budget = Budget(1000, 100)
    ctx = build(turns, budget)
    assert ctx.last_turn_truncated and ctx.recent_turns >= 1
    answer = ctx.messages[-2]["content"]
    assert answer.endswith("[…]")
    assert words(answer) + O + 1 + O <= int(budget.last_turn_max_share * available(budget))


def test_prompt_too_long_raises():
    with pytest.raises(PromptTooLongError):
        build([], Budget(100, 50), prompt=" ".join(["w"] * 100))


def test_fragments_get_the_space_left_by_recent_turns():
    turns = [turn(i, size=100) for i in range(1, 13)]
    budget = Budget(600, 100)                                   # available 430, recent cap 258
    fragments = [Fragment(chunk_id=7, turn=2, ord=0, content="hash password", header="login api"),
                 Fragment(chunk_id=9, turn=11, ord=0, content="already whole")]
    ctx = build(turns, budget, fragments=fragments)
    # 2 turns fit the recent cap (216 <= 258 < 324); the fragment is small, so step 4 brings
    # back turn 10 whole in the space left (216 + 108 + fragment <= 430)
    assert ctx.recent_turns == 3
    assert ctx.fragment_ids == [7]                              # turn 11 is included whole
    assert roles(ctx)[:2] == ["system", "system"]
    assert "[turn 2] Question: login api" in ctx.messages[1]["content"]


def test_left_over_space_brings_back_whole_turns_and_drops_their_fragments():
    turns = [turn(i, size=20) for i in range(1, 5)]             # 28 tokens each
    budget = Budget(400, 50)                                    # available 400-50-10-40 = 300
    fragments = [Fragment(chunk_id=1, turn=1, ord=0, content="old bit")]
    ctx = build(turns, budget, fragments=fragments)
    assert ctx.recent_turns == 4 and ctx.fragment_ids == []    # turn 1 fits whole after all


def test_render_groups_by_turn_and_marks_gaps():
    text = render_excerpts([
        Fragment(3, turn=5, ord=4, content="c"),
        Fragment(1, turn=2, ord=0, content="a", header="how to log in"),
        Fragment(2, turn=2, ord=2, content="b", header="how to log in"),
        Fragment(4, turn=5, ord=1, content="d", header="Error comes from the token", header_from_answer=True),
    ])
    assert text.splitlines() == [
        "Relevant excerpts from earlier in this conversation:", "",
        "[turn 2] Question: how to log in", "a", "[…]", "b", "",
        "[turn 5] Topic: Error comes from the token", "d", "[…]", "c",
    ]


def test_truncate_to():
    assert truncate_to("a b c", 5, words) == "a b c"
    cut = truncate_to("a b c d e f", 4, words)
    assert cut.endswith("[…]") and words(cut) <= 4
