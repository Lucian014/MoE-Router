"""Session routing rules (src/router/session_routing.py), with fake classifier outputs."""
from dataclasses import dataclass

import pytest

from router.session_routing import RoutingRules, SessionState, decide, is_ambiguous, is_short

RULES = RoutingRules()
CODING_SESSION = SessionState(domain="coding", domain_confidence=0.91, complexity_score=0.8)


@dataclass
class Domain:
    category: str
    confidence: float
    probabilities: dict


@dataclass
class Complexity:
    level: str
    score: float


def classifier(domain_probs, complexity_score=0.2):
    calls = []

    def classify(prompt):
        calls.append(prompt)
        top = max(domain_probs, key=domain_probs.get)
        return (Domain(top, domain_probs[top], domain_probs),
                Complexity("high" if complexity_score >= 0.5 else "low", complexity_score))

    return classify, calls


CLEAR_REWRITE = {"text-rewrite": 0.91, "coding": 0.02, "math": 0.07}
FLAT = {"open-qa": 0.17, "brainstorming": 0.16, "summarization": 0.16, "coding": 0.11}


def test_first_turn_uses_the_classifiers_as_they_are():
    classify, _ = classifier(FLAT)
    d = decide("scrie o funcție Python care sortează o listă", None, classify, RULES)
    assert (d.domain, d.inherited, d.reason) == ("open-qa", False, "first-turn")


def test_short_follow_up_inherits_without_classifying():
    classify, calls = classifier(CLEAR_REWRITE)          # the confidently wrong case seen in the cluster
    d = decide("fă-o descrescătoare", CODING_SESSION, classify, RULES)
    assert calls == []
    assert (d.domain, d.complexity_level, d.inherited, d.reason) == ("coding", "high", True, "short-prompt")
    assert d.predicted_domain is None


def test_longer_prompts_are_classified_whatever_their_first_word():
    classify, calls = classifier({"coding": 0.9, "math": 0.05})
    d = decide("și acum scrie-o în Rust te rog", CODING_SESSION, classify, RULES)
    assert (d.domain, d.reason, d.inherited, len(calls)) == ("coding", "classified", False, 1)


def test_ambiguous_domain_inherits_both_and_keeps_the_prediction():
    classify, calls = classifier(FLAT, complexity_score=0.1)
    d = decide("cum ar arăta asta într-un context mai general de lucru?", CODING_SESSION, classify, RULES)
    assert len(calls) == 1
    assert (d.domain, d.complexity_level, d.inherited, d.reason) == ("coding", "high", True, "ambiguous-domain")
    assert (d.predicted_domain, d.predicted_complexity) == ("open-qa", 0.1)


def test_clear_new_topic_switches_domain():
    classify, _ = classifier({"math": 0.9, "coding": 0.05}, complexity_score=0.3)
    d = decide("calculează integrala lui x pătrat între 0 și 3", CODING_SESSION, classify, RULES)
    assert (d.domain, d.complexity_level, d.inherited, d.reason) == ("math", "low", False, "classified")


def test_short_prompt_without_session_state_is_classified():
    classify, calls = classifier(CLEAR_REWRITE)
    empty = SessionState(domain=None, domain_confidence=None, complexity_score=None)
    d = decide("continua", empty, classify, RULES)
    assert calls == ["continua"] and not d.inherited


@pytest.mark.parametrize("prompt, expected", [
    ("continua", True),
    ("ok mersi", True),
    ("fă-o descrescătoare", True),
    ("make it descending", True),
    ("and now in Rust please", False),
    ("scrie o funcție Python care sortează o listă", False),
])
def test_is_short(prompt, expected):
    assert is_short(prompt, RULES) is expected


@pytest.mark.parametrize("probs, expected", [
    ({"a": 0.91, "b": 0.05}, False),
    ({"a": 0.45, "b": 0.10}, True),          # below min_confidence
    ({"a": 0.55, "b": 0.45}, True),          # top two too close
    ({}, True),
])
def test_is_ambiguous(probs, expected):
    assert is_ambiguous(probs, RULES) is expected
