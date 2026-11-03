"""
router/laya_classifier.py -- drop-in replacement for router/classifier.py backed by a
fine-tuned Laya checkpoint. One forward pass answers BOTH questions: task type and complexity.

Same public interface as before (load_classifier, classify, CLASSIFIER_DEVICE and a
ClassificationResult with task_type_1/2, confidences and margin), plus complexity fields,
so existing router code keeps working and can start using complexity when ready.

Env:
  LAYA_CHECKPOINT   path or HF repo of the fine-tuned model (default: checkpoints/laya-router)
  CLASSIFIER_DEVICE cpu | cuda | mps (default: cpu)
"""
import os
from dataclasses import dataclass, field

import laya

from laya_task_schema import COMPLEXITY_QID, COMPLEXITY_TIERS, QUESTIONS, TASK_QID

LAYA_CHECKPOINT = os.getenv("LAYA_CHECKPOINT", "checkpoints/laya-router")
CLASSIFIER_DEVICE = os.getenv("CLASSIFIER_DEVICE", "cpu")

_AGENT = None


@dataclass
class ClassificationResult:
    task_type_1: str
    task_type_1_confidence: float
    task_type_2: str
    task_type_2_confidence: float
    margin: float
    complexity_tier: str = "medium"          # "low" | "medium" | "high"
    complexity_score: float = 1.0            # expected level, 0.0 (low) .. 2.0 (high)
    complexity_probabilities: dict = field(default_factory=dict)


def load_classifier() -> None:
    global _AGENT
    print(f"[*] Loading Laya decision model from '{LAYA_CHECKPOINT}' on {CLASSIFIER_DEVICE}...")
    _AGENT = laya.Agent(LAYA_CHECKPOINT, device=CLASSIFIER_DEVICE)
    print(f"[OK] Laya ready. Questions: {list(QUESTIONS)}")


def classify(prompt: str) -> ClassificationResult:
    if _AGENT is None:
        raise RuntimeError("Classifier not loaded. Call load_classifier() first.")
    answers = _AGENT.predict(prompt, QUESTIONS)["answers"]

    task = answers[TASK_QID]
    ranked = sorted(task["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
    (l1, p1), (l2, p2) = ranked[0], ranked[1]

    cx = answers[COMPLEXITY_QID]
    cx_probs = {COMPLEXITY_TIERS[int(k)]: v for k, v in cx["probabilities"].items()}
    tier = COMPLEXITY_TIERS[min(len(COMPLEXITY_TIERS) - 1, int(round(cx["score"])))]

    return ClassificationResult(l1, float(p1), l2, float(p2), float(p1 - p2),
                                tier, float(cx["score"]), cx_probs)