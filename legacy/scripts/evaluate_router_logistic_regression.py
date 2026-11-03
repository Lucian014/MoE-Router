#!/usr/bin/env python3
"""
evaluate_router_logistic_regression.py

Same evaluation harness as evaluate_router.py, but with the task
classification step swapped out: instead of the NVIDIA
DeBERTa classifier (router.classifier.classify), this loads the
multinomial logistic regression model produced by train_classifier.py
and uses IT to get task_type_1/2, confidences, and margin.

Everything downstream is untouched on purpose, so results are directly
comparable to evaluate_router.py's output on the same test-prompts file:

  - candidate filtering by task_type (from the YAML catalog)
  - embedding similarity scoring within the candidate pool
    (same 40% description / 60% closest-example formula as select_model())
  - General Purpose fallback when task_type == "Other"/ambiguous, or
    margin < --margin-threshold, or no candidate declares the task_type
  - accuracy summary + confusion matrix + closest-correct-calls report

Usage:
    python evaluate_router_logistic_regression.py \\
        --catalog src/catalog/models.yaml \\
        --test-prompts test_prompts_bulk.csv \\
        --classifier router/task_classifier.joblib \\
        --output results_lr.csv

    # to compare head-to-head against the NVIDIA classifier on the exact
    # same test set:
    python evaluate_router.py --catalog ... --test-prompts SAME_FILE.csv --output results_nvidia.csv
    python evaluate_router_logistic_regression.py --catalog ... --test-prompts SAME_FILE.csv --output results_lr.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import yaml
from sentence_transformers import SentenceTransformer


# =============================================================================
# Task classifier: multinomial logistic regression (replaces
# router.classifier.classify for the purposes of this eval script)
# =============================================================================

@dataclass
class ClassificationResult:
    task_type_1: str
    task_type_1_confidence: float
    task_type_2: str
    task_type_2_confidence: float
    margin: float


class LogisticRegressionTaskClassifier:
    """Thin wrapper around the .joblib produced by train_classifier.py,
    exposing the same shape of result as router.classifier.classify()
    so RouterEvalEngine.route() below needs zero changes to consume it."""

    def __init__(self, classifier_path: str, shared_embedder: SentenceTransformer | None = None):
        if not Path(classifier_path).exists():
            raise FileNotFoundError(
                f"Classifier not found at '{classifier_path}'. Train it first with "
                f"train_classifier.py (--output {classifier_path})."
            )
        bundle = joblib.load(classifier_path)
        self.clf = bundle["classifier"]
        self.embedding_model_name = bundle["embedding_model"]

        if shared_embedder is not None:
            # Reuse the same embedder instance the router's own
            # select_model() step uses, IF it's the same model -- avoids
            # loading two copies of a multi-GB model into memory.
            self.embedder = shared_embedder
            self._owns_embedder = False
        else:
            print(f"[*] Loading embedding model '{self.embedding_model_name}' (CPU) for classification...")
            self.embedder = SentenceTransformer(self.embedding_model_name, device="cpu")
            self._owns_embedder = True

    def classify(self, prompt: str) -> ClassificationResult:
        embedding = self.embedder.encode([prompt], convert_to_numpy=True, show_progress_bar=False)
        probs = self.clf.predict_proba(embedding)[0]
        order = np.argsort(probs)[::-1]
        top1_idx, top2_idx = order[0], order[1]
        return ClassificationResult(
            task_type_1=str(self.clf.classes_[top1_idx]),
            task_type_1_confidence=float(probs[top1_idx]),
            task_type_2=str(self.clf.classes_[top2_idx]),
            task_type_2_confidence=float(probs[top2_idx]),
            margin=float(probs[top1_idx] - probs[top2_idx]),
        )


# =============================================================================
# Catalog loading (YAML) -- identical to evaluate_router.py
# =============================================================================

def load_catalog_from_yaml(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    all_models = data.get("models", [])
    if not all_models:
        raise ValueError(f"No models found in catalog file: {path}")

    catalog = [entry for entry in all_models if entry.get("enabled", True)]
    if not catalog:
        raise ValueError(f"No *enabled* models found in catalog file: {path}")

    return catalog


def resolve_general_purpose_model(catalog: list[dict], override: str | None) -> str:
    if override:
        model_ids = {entry["model_id"] for entry in catalog}
        if override not in model_ids:
            raise ValueError(
                f"--general-purpose-model override='{override}' is not present "
                f"in the enabled catalog {sorted(model_ids)}."
            )
        return override

    defaults = [entry["model_id"] for entry in catalog if entry.get("is_default")]
    if len(defaults) == 0:
        raise ValueError(
            "No model in the catalog has is_default: true, and no "
            "--general-purpose-model override was given."
        )
    if len(defaults) > 1:
        raise ValueError(f"Multiple models have is_default: true: {defaults}. Exactly one is required.")

    return defaults[0]


# =============================================================================
# Same similarity math as router.py / evaluate_router.py (unchanged on
# purpose, so results are comparable apples-to-apples against the NVIDIA
# classifier run)
# =============================================================================

def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms


class RouterEvalEngine:
    """Identical to evaluate_router.py's RouterEvalEngine, except `route()`
    calls self.classifier.classify(prompt) -- the injected logistic
    regression classifier -- instead of router.classifier.classify()."""

    def __init__(
        self,
        embedder: SentenceTransformer,
        classifier: LogisticRegressionTaskClassifier,
        description_weight: float,
        example_weight: float,
        margin_threshold: float,
        general_purpose_model: str,
    ):
        self.embedder = embedder
        self.classifier = classifier
        self.description_weight = description_weight
        self.example_weight = example_weight
        self.margin_threshold = margin_threshold
        self.general_purpose_model = general_purpose_model

        self.model_names: list[str] = []
        self.model_task_types: list[set[str]] = []
        self.description_embeddings: np.ndarray | None = None
        self.example_embeddings: np.ndarray | None = None

    def load(self, catalog: list[dict]) -> None:
        self.model_names = [entry["model_id"] for entry in catalog]
        self.model_task_types = [set(entry.get("task_types", [])) for entry in catalog]

        descriptions = [entry.get("description", "") for entry in catalog]

        raw_description_embeddings = self.embedder.encode(
            descriptions, convert_to_numpy=True, show_progress_bar=False
        )
        self.description_embeddings = _normalize(raw_description_embeddings).astype(np.float32)

        max_examples = max((len(entry.get("examples", [])) for entry in catalog), default=0)
        embedding_dim = self.description_embeddings.shape[1]

        if max_examples == 0:
            self.example_embeddings = np.empty((len(catalog), 0, embedding_dim), dtype=np.float32)
            return

        example_embeddings = np.zeros((len(catalog), max_examples, embedding_dim), dtype=np.float32)

        for model_idx, entry in enumerate(catalog):
            examples = entry.get("examples", [])
            if not examples:
                continue
            raw_examples = self.embedder.encode(examples, convert_to_numpy=True, show_progress_bar=False)
            example_embeddings[model_idx, : len(examples), :] = _normalize(raw_examples).astype(np.float32)

        self.example_embeddings = example_embeddings

    def _candidate_indices_for_task_type(self, task_type: str) -> list[int]:
        """All enabled model indices whose task_types contains `task_type`.

        Matches router.py's real _candidate_indices_for_task_type() exactly:
        the General Purpose model is NOT excluded from the candidate pool.
        It competes on cosine similarity like any other model that shares
        the task_type -- it only ever wins "automatically" via the
        margin-threshold ambiguity check in route(), never by being the
        sole survivor of an artificial exclusion here.

        (The original evaluate_router.py had an extra `non_default = [...]`
        filter here that silently removed the General Purpose model from
        the pool whenever it shared a task_type with another model -- e.g.
        llama3.2:3b vs mightykatun/qwen2.5-math:1.5b under "open-qa". That
        made every non-ambiguous open-qa prediction default to the OTHER
        model by construction, regardless of prompt content. Removed here
        so eval numbers reflect what router.py actually does in production.)
        """
        return [idx for idx, task_types in enumerate(self.model_task_types) if task_type in task_types]

    def _similarity_over_candidates(self, prompt_embedding: np.ndarray, candidate_indices: list[int]) -> dict:
        candidate_description_embeddings = self.description_embeddings[candidate_indices]
        candidate_example_embeddings = self.example_embeddings[candidate_indices]

        description_similarities = candidate_description_embeddings @ prompt_embedding

        if candidate_example_embeddings.shape[1] > 0:
            example_similarities = np.einsum("med,d->me", candidate_example_embeddings, prompt_embedding)
            closest_example_similarities = np.max(example_similarities, axis=1)
        else:
            closest_example_similarities = np.zeros(len(candidate_indices), dtype=np.float32)

        routing_score = (
            self.description_weight * description_similarities
            + self.example_weight * closest_example_similarities
        )

        order = np.argsort(routing_score)[::-1]
        best_local = int(order[0])
        best_global = candidate_indices[best_local]

        if len(order) > 1:
            second_local = int(order[1])
            second_global = candidate_indices[second_local]
            runner_up_model = self.model_names[second_global]
            similarity_margin = float(routing_score[best_local] - routing_score[second_local])
        else:
            runner_up_model = None
            similarity_margin = None

        return {
            "selected_model": self.model_names[best_global],
            "routing_score": float(routing_score[best_local]),
            "description_sim": float(description_similarities[best_local]),
            "example_sim": float(closest_example_similarities[best_local]),
            "runner_up_model": runner_up_model,
            "similarity_margin": similarity_margin,
            "candidate_pool_size": len(candidate_indices),
        }

    def route(self, prompt: str) -> dict:
        """Full pipeline: LR classify -> ambiguity check -> filter by
        task_type -> similarity on candidate pool -> or General Purpose
        fallback. Mirrors router.py's route_message() exactly, just with
        the classifier swapped."""
        classification = self.classifier.classify(prompt)
        task_type = classification.task_type_1
        task_type_confidence = classification.task_type_1_confidence
        task_type_margin = classification.margin

        is_ambiguous = task_type == "Other" or task_type_margin < self.margin_threshold

        base_result = {
            "task_type": task_type,
            "task_type_confidence": task_type_confidence,
            "task_type_margin": task_type_margin,
        }

        if is_ambiguous:
            return {
                **base_result,
                "selected_model": self.general_purpose_model,
                "is_fallback": True,
                "fallback_reason": "ambiguous_or_other",
                "routing_score": None,
                "description_sim": None,
                "example_sim": None,
                "runner_up_model": None,
                "similarity_margin": None,
                "candidate_pool_size": None,
            }

        candidate_indices = self._candidate_indices_for_task_type(task_type)

        if not candidate_indices:
            return {
                **base_result,
                "selected_model": self.general_purpose_model,
                "is_fallback": True,
                "fallback_reason": "no_candidate_for_task_type",
                "routing_score": None,
                "description_sim": None,
                "example_sim": None,
                "runner_up_model": None,
                "similarity_margin": None,
                "candidate_pool_size": None,
            }

        prompt_embedding = self.embedder.encode([prompt], convert_to_numpy=True, show_progress_bar=False)
        prompt_embedding = _normalize(prompt_embedding).astype(np.float32)[0]

        similarity_result = self._similarity_over_candidates(prompt_embedding, candidate_indices)

        return {
            **base_result,
            **similarity_result,
            "is_fallback": False,
            "fallback_reason": None,
        }


# =============================================================================
# Test prompt loading -- identical to evaluate_router.py (prompt,expected_model)
# =============================================================================

def load_test_prompts(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)

        if reader.fieldnames is None or not {"prompt", "expected_model"}.issubset(set(reader.fieldnames)):
            raise ValueError(
                f"Test prompts file '{path}' must have a header with at least "
                f"'prompt' and 'expected_model' columns. Found: {reader.fieldnames}"
            )

        rows = [row for row in reader if row.get("prompt", "").strip()]

    if not rows:
        raise ValueError(f"No test prompts found in '{path}'.")

    return rows


# =============================================================================
# Evaluation loop -- identical to evaluate_router.py
# =============================================================================

def run_evaluation(engine: RouterEvalEngine, test_rows: list[dict]) -> list[dict]:
    results = []

    for row in test_rows:
        prompt = row["prompt"]
        expected_model = row["expected_model"].strip()

        outcome = engine.route(prompt)
        correct = outcome["selected_model"] == expected_model

        def _round_or_none(value):
            return round(value, 4) if value is not None else None

        results.append(
            {
                "prompt": prompt,
                "expected_model": expected_model,
                "selected_model": outcome["selected_model"],
                "correct": correct,
                "is_fallback": outcome["is_fallback"],
                "fallback_reason": outcome["fallback_reason"],
                "task_type": outcome["task_type"],
                "task_type_confidence": _round_or_none(outcome["task_type_confidence"]),
                "task_type_margin": _round_or_none(outcome["task_type_margin"]),
                "routing_score": _round_or_none(outcome["routing_score"]),
                "description_sim": _round_or_none(outcome["description_sim"]),
                "example_sim": _round_or_none(outcome["example_sim"]),
                "runner_up_model": outcome["runner_up_model"],
                "similarity_margin": _round_or_none(outcome["similarity_margin"]),
                "candidate_pool_size": outcome["candidate_pool_size"],
            }
        )

    return results


def write_results_csv(results: list[dict], output_path: str) -> None:
    fieldnames = [
        "prompt", "expected_model", "selected_model", "correct", "is_fallback", "fallback_reason",
        "task_type", "task_type_confidence", "task_type_margin", "routing_score",
        "description_sim", "example_sim", "runner_up_model", "similarity_margin", "candidate_pool_size",
    ]
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def print_summary(results: list[dict]) -> None:
    total = len(results)
    correct = sum(1 for r in results if r["correct"])
    accuracy = correct / total if total else 0.0

    print("\n" + "=" * 70)
    print(f"ACCURACY (logistic regression classifier): {correct}/{total} ({accuracy:.1%})")
    print("=" * 70)

    fallback_count = sum(1 for r in results if r["is_fallback"])
    if fallback_count:
        fallback_correct = sum(1 for r in results if r["is_fallback"] and r["correct"])
        print(
            f"\nFALLBACK TO GENERAL PURPOSE: {fallback_count}/{total} prompt(s) "
            f"({fallback_correct} of those matched expected_model)."
        )
        by_reason: dict[str, int] = defaultdict(int)
        for r in results:
            if r["is_fallback"]:
                by_reason[r["fallback_reason"]] += 1
        for reason, count in by_reason.items():
            print(f"    {reason}: {count}")

    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        confusion[r["expected_model"]][r["selected_model"]] += 1

    print("\nCONFUSION SUMMARY (only rows with mistakes shown):\n")
    any_mistakes = False
    for expected_model in sorted(confusion.keys()):
        row = confusion[expected_model]
        total_for_model = sum(row.values())
        correct_for_model = row.get(expected_model, 0)

        if correct_for_model == total_for_model:
            continue

        any_mistakes = True
        print(f"  Expected '{expected_model}' ({correct_for_model}/{total_for_model} correct):")
        for selected_model, count in sorted(row.items(), key=lambda kv: -kv[1]):
            if selected_model == expected_model:
                continue
            print(f"      -> misrouted to '{selected_model}': {count} time(s)")

    if not any_mistakes:
        print("  No mistakes -- every model routed 100% correctly on this test set.")

    single_candidate_count = sum(1 for r in results if not r["is_fallback"] and r["candidate_pool_size"] == 1)
    if single_candidate_count:
        print(
            f"\n[*] {single_candidate_count}/{total} non-fallback prediction(s) had exactly one "
            f"candidate for their task_type -- no real competition happened."
        )

    close_calls = sorted(
        (r for r in results if r["correct"] and not r["is_fallback"] and r["similarity_margin"] is not None),
        key=lambda r: r["similarity_margin"],
    )[:5]

    if close_calls:
        print("\nCLOSEST CORRECT CALLS (small margin = fragile, worth reviewing):\n")
        for r in close_calls:
            print(
                f"  margin={r['similarity_margin']:.4f}  '{r['prompt'][:60]}' "
                f"-> {r['selected_model']} (runner-up: {r['runner_up_model']})"
            )

    print()


# =============================================================================
# CLI
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the semantic router (logistic regression classifier) against labeled test prompts.")
    parser.add_argument("--catalog", default="src/catalog/models.yaml", help="Path to the models YAML catalog.")
    parser.add_argument("--test-prompts", default="test_prompts.csv", help="Path to the labeled test prompts CSV (prompt,expected_model).")
    parser.add_argument("--output", default="results_lr.csv", help="Path to write the results CSV.")
    parser.add_argument("--classifier", default="router/task_classifier.joblib", help="Path to the .joblib from train_classifier.py.")
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL_NAME", "BAAI/bge-m3"),
        help="Embedder for the router's own candidate-selection step (description/example similarity). "
             "Should match router.py's EMBEDDING_MODEL_NAME.",
    )
    parser.add_argument("--description-weight", type=float, default=0.4)
    parser.add_argument("--example-weight", type=float, default=0.6)
    parser.add_argument("--margin-threshold", type=float, default=float(os.getenv("CLASSIFIER_MARGIN_THRESHOLD", "0.05")))
    parser.add_argument("--general-purpose-model", default=os.getenv("GENERAL_PURPOSE_MODEL"))
    args = parser.parse_args()

    if not Path(args.catalog).exists():
        sys.exit(f"[!] Catalog file not found: {args.catalog}")
    if not Path(args.test_prompts).exists():
        sys.exit(f"[!] Test prompts file not found: {args.test_prompts}")

    print(f"[*] Loading catalog from '{args.catalog}'...")
    catalog = load_catalog_from_yaml(args.catalog)

    general_purpose_model = resolve_general_purpose_model(catalog, args.general_purpose_model)
    print(f"[OK] General Purpose model: '{general_purpose_model}'")

    print(f"[*] Loading test prompts from '{args.test_prompts}'...")
    test_rows = load_test_prompts(args.test_prompts)
    print(f"[*] {len(test_rows)} test prompt(s) loaded.")

    print(f"[*] Loading embedding model '{args.embedding_model}' (CPU)...")
    embedder = SentenceTransformer(args.embedding_model, device="cpu")

    print(f"[*] Loading trained logistic regression task classifier from '{args.classifier}'...")
    classifier = LogisticRegressionTaskClassifier(args.classifier, shared_embedder=embedder)
    if classifier.embedding_model_name != args.embedding_model:
        print(
            f"[!] WARNING: classifier was trained with embedding model "
            f"'{classifier.embedding_model_name}', but this run is using "
            f"'{args.embedding_model}' for candidate selection. They should "
            f"match --embedding-model to --embedding-model used in train_classifier.py, "
            f"or classification and candidate-selection are comparing apples to oranges."
        )

    engine = RouterEvalEngine(
        embedder=embedder,
        classifier=classifier,
        description_weight=args.description_weight,
        example_weight=args.example_weight,
        margin_threshold=args.margin_threshold,
        general_purpose_model=general_purpose_model,
    )
    engine.load(catalog)
    print(f"[OK] Cached embeddings for {len(engine.model_names)} model(s): {engine.model_names}")

    print("[*] Running evaluation...")
    results = run_evaluation(engine, test_rows)

    write_results_csv(results, args.output)
    print(f"[OK] Results written to '{args.output}'")

    print_summary(results)


if __name__ == "__main__":
    main()