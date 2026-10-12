"""
router/classifier.py -- the router's two INDEPENDENT Laya classifiers behind one interface.

    load_classifiers()                        once, at router start-up
    predict_domain(prompt)     -> DomainPrediction       (8 categories, all probabilities)
    predict_complexity(prompt) -> ComplexityPrediction   (low / high, score = P(high))

The router talks to these functions, never to Laya directly (CLAUDE.md: classifiers are
interchangeable). The question text lives in domain_schema.py / complexity_schema.py and must be
byte-identical to the one used in training -- copy those files from scripts/laya/, never edit them here.

Env:
  DOMAIN_CHECKPOINT      local folder or HF repo  (default: router/checkpoints/laya-domain-v4,
                                                    else Lucian04/laya-domain-v4)
  COMPLEXITY_CHECKPOINT  local folder or HF repo  (default: router/checkpoints/laya-complexity-v2,
                                                    else Lucian04/laya-complexity-v2)
  CLASSIFIER_DEVICE      cpu | cuda               (default: cpu)
  COMPLEXITY_THRESHOLD   "high" when P(high) >= this (default 0.5; 0.3-0.6 give the same routing)
  HF_TOKEN               only when loading the private repos from the Hub
"""
import os
from dataclasses import dataclass
from pathlib import Path

try:                                    # run as package (python -m router.router) ...
    from router import complexity_schema, domain_schema
except ImportError:                     # ... or from inside src/router/
    import complexity_schema
    import domain_schema

_HERE = Path(__file__).resolve().parent


def _default_checkpoint(local_name: str, hub_repo: str) -> str:
    local = _HERE / "checkpoints" / local_name
    return str(local) if local.is_dir() else hub_repo


DOMAIN_CHECKPOINT = os.getenv("DOMAIN_CHECKPOINT") or _default_checkpoint(
    "laya-domain-v4", "Lucian04/laya-domain-v4")
COMPLEXITY_CHECKPOINT = os.getenv("COMPLEXITY_CHECKPOINT") or _default_checkpoint(
    "laya-complexity-v2", "Lucian04/laya-complexity-v2")
CLASSIFIER_DEVICE = os.getenv("CLASSIFIER_DEVICE", "cpu")
COMPLEXITY_THRESHOLD = float(os.getenv("COMPLEXITY_THRESHOLD", "0.5"))

CATEGORIES = list(domain_schema.LABELS)
LEVELS = list(complexity_schema.LABELS)


@dataclass
class DomainPrediction:
    category: str
    confidence: float        # its probability (logged only: NOT used for routing)
    probabilities: dict      # {category: probability} for all 8 categories


@dataclass
class ComplexityPrediction:
    level: str               # "low" | "high"
    score: float             # P(high), 0..1
    probabilities: dict      # {"low": p, "high": p}


class _LayaClassifier:
    def __init__(self, checkpoint: str, schema):
        import laya
        self.schema = schema
        print(f"[*] Loading {schema.QID} classifier from '{checkpoint}' on {CLASSIFIER_DEVICE}...")
        self.agent = laya.Agent(checkpoint, device=CLASSIFIER_DEVICE)

    def answer(self, prompt: str) -> dict:
        return self.agent.predict(prompt, self.schema.QUESTIONS)["answers"][self.schema.QID]


_DOMAIN: _LayaClassifier | None = None
_COMPLEXITY: _LayaClassifier | None = None


def load_classifiers() -> None:
    global _DOMAIN, _COMPLEXITY
    _DOMAIN = _LayaClassifier(DOMAIN_CHECKPOINT, domain_schema)
    _COMPLEXITY = _LayaClassifier(COMPLEXITY_CHECKPOINT, complexity_schema)
    predict_domain("warm-up")
    predict_complexity("warm-up")
    print(f"[OK] classifiers ready: domain {CATEGORIES} | complexity {LEVELS} "
          f"(threshold {COMPLEXITY_THRESHOLD})")


def predict_domain(prompt: str) -> DomainPrediction:
    if _DOMAIN is None:
        raise RuntimeError("Call load_classifiers() first.")
    probs = {c: float(p) for c, p in _DOMAIN.answer(prompt)["probabilities"].items()}
    category = max(probs, key=probs.get)
    return DomainPrediction(category, probs[category], probs)


def predict_complexity(prompt: str) -> ComplexityPrediction:
    if _COMPLEXITY is None:
        raise RuntimeError("Call load_classifiers() first.")
    a = _COMPLEXITY.answer(prompt)
    probs = {LEVELS[int(k)]: float(v) for k, v in a["probabilities"].items()}
    score = float(a["score"])
    return ComplexityPrediction(LEVELS[int(score >= COMPLEXITY_THRESHOLD)], score, probs)