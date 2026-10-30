"""
router/session_routing.py -- which domain and complexity a prompt is routed on, given its session
(docs/rag-planv2.md 4.1). Pure: the classifiers are passed in, so the rules are testable without them.

The classifiers only ever see the current prompt, as in training. The session supplies the context.
The rules are language-independent (no word lists):
  1. short prompt (at most short_prompt_max_words words) and the session has a state
       -> inherit domain + complexity, without classifying. On such prompts the domain classifier
          can be confidently wrong ("continua" -> summarization 0.91)
  2. otherwise classify the prompt
  3. ambiguous domain (top probability < min_confidence, or top two closer than min_margin)
     and the session has a state -> inherit BOTH (an ambiguous prompt is most likely a follow-up)
  4. no state yet (first turn): the classifiers' answer as it is
"""
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionState:
    """The effective routing of the session's last turn (the `sessions` row)."""
    domain: str | None
    domain_confidence: float | None
    complexity_score: float | None

    @property
    def usable(self) -> bool:
        return self.domain is not None and self.complexity_score is not None


@dataclass(frozen=True)
class RoutingRules:
    short_prompt_max_words: int = 3
    min_confidence: float = 0.5
    min_margin: float = 0.15
    complexity_threshold: float = 0.5      # P(high) >= this -> "high", like the classifier


@dataclass(frozen=True)
class RoutingDecision:
    domain: str
    domain_confidence: float
    complexity_score: float
    complexity_level: str
    inherited: bool
    reason: str                            # classified | first-turn | short-prompt | ambiguous-domain
    predicted_domain: str | None = None    # None when the classifiers were skipped
    predicted_complexity: float | None = None
    domain_prediction: object = None       # the classifiers' outputs, for the routing metadata
    complexity_prediction: object = None


def is_short(prompt: str, rules: RoutingRules) -> bool:
    return len(prompt.split()) <= rules.short_prompt_max_words


def is_ambiguous(probabilities: dict[str, float], rules: RoutingRules) -> bool:
    top = sorted(probabilities.values(), reverse=True)
    if not top:
        return True
    second = top[1] if len(top) > 1 else 0.0
    return top[0] < rules.min_confidence or top[0] - second < rules.min_margin


def decide(prompt: str, state: SessionState | None, classify: Callable[[str], tuple],
           rules: RoutingRules) -> RoutingDecision:
    """`classify(prompt)` -> (domain prediction, complexity prediction), as router.classify()."""
    has_state = state is not None and state.usable

    if has_state and is_short(prompt, rules):
        return _inherit(state, rules, "short-prompt")

    domain, complexity = classify(prompt)
    if has_state and is_ambiguous(domain.probabilities, rules):
        return _inherit(state, rules, "ambiguous-domain", domain, complexity)

    return RoutingDecision(
        domain=domain.category,
        domain_confidence=domain.confidence,
        complexity_score=complexity.score,
        complexity_level=complexity.level,
        inherited=False,
        reason="classified" if has_state else "first-turn",
        predicted_domain=domain.category,
        predicted_complexity=complexity.score,
        domain_prediction=domain,
        complexity_prediction=complexity,
    )


def _inherit(state: SessionState, rules: RoutingRules, reason: str,
             domain=None, complexity=None) -> RoutingDecision:
    level = "high" if state.complexity_score >= rules.complexity_threshold else "low"
    return RoutingDecision(
        domain=state.domain,
        domain_confidence=state.domain_confidence if state.domain_confidence is not None else 0.0,
        complexity_score=state.complexity_score,
        complexity_level=level,
        inherited=True,
        reason=reason,
        predicted_domain=domain.category if domain else None,
        predicted_complexity=complexity.score if complexity else None,
        domain_prediction=domain,
        complexity_prediction=complexity,
    )
