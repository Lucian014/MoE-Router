#!/usr/bin/env python3
"""
evaluate_router.py

Standalone evaluation harness for the semantic router.

  - loads the model catalog straight from a YAML file
    (e.g. src/catalog/models.yaml), instead of the `models` DB table
  - loads test prompts (prompt + expected_model) from a CSV file
  - runs the FULL pipeline for every test prompt: classify -> filter by
    task_type -> embedding similarity on the candidate pool only -> or
    fall back to the General Purpose model if the classifier is ambiguous
    / returns "Other" / no candidate supports the detected task_type
  - writes a results CSV (prompt, expected, selected, correct, task_type,
    task_type_confidence, task_type_margin, is_fallback, scores, margin)
  - prints an accuracy summary and a confusion matrix to the console

Usage:
    python evaluate_router.py \\
        --catalog src/catalog/models.yaml \\
        --test-prompts test_prompts.csv \\
        --output results.csv

    # optional (should match the live router's env vars / defaults):
    #   --embedding-model BAAI/bge-m3
    #   --description-weight 0.4
    #   --example-weight 0.6
    #   --margin-threshold 0.10
    #   --general-purpose-model some-model-id   (overrides is_default in YAML)
    #   --classifier-device cpu

"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from sentence_transformers import SentenceTransformer

# Same pattern as scripts/seed_models.py and scripts/validate_task_types.py:
# add `src/` to sys.path so top-level packages (router, database, shared)
# are importable regardless of where this script is invoked from. There is
# no `src` package at runtime.
ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from router.classifier import load_classifier, classify  # noqa: E402


# =============================================================================
# Catalog loading (YAML instead of DB)
# =============================================================================

def load_catalog_from_yaml(path: str) -> list[dict]:
    """Load the model catalog from a YAML file with a top-level `models:`
    list. Mirrors ModelRepository.list_enabled(): only enabled models are
    returned, same as what the live router would see."""
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
    """Mirrors router.py's load_router_catalog(): use the override if given,
    otherwise the single catalog entry with is_default: true, otherwise
    fail loudly instead of guessing."""
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
            "--general-purpose-model override was given. Mark exactly one "
            "model as default in the YAML, or pass the override flag."
        )
    if len(defaults) > 1:
        raise ValueError(f"Multiple models have is_default: true: {defaults}. Exactly one is required.")

    return defaults[0]


# =============================================================================
# Same math as router.py (kept identical on purpose, so eval results are
# directly comparable to what the live router would do)
# =============================================================================

def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalize row vectors so a plain dot product == cosine similarity."""
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    norms[norms == 0] = 1e-12
    return vectors / norms


class RouterEvalEngine:
    """Mirrors the full router.py pipeline (classify -> filter -> similarity
    -> General Purpose fallback), decoupled from DB/MQ."""

    def __init__(
        self,
        embedder: SentenceTransformer,
        description_weight: float,
        example_weight: float,
        margin_threshold: float,
        general_purpose_model: str,
        default_model_penalty: float = 0.03,
        reasoning_bonus: float = 0.0,
        domain_bonus: float = 0.0,
    ):
        self.embedder = embedder
        self.description_weight = description_weight
        self.example_weight = example_weight
        self.margin_threshold = margin_threshold
        self.general_purpose_model = general_purpose_model

        # Small bias subtracted from the General Purpose model's routing_score
        # when it competes against specialists on a shared task_type (e.g.
        # "Open QA" claimed by both the default chat model and the math
        # model). See conversation history: excluding the default outright
        # broke its OWN legitimate "Open QA" traffic (general chat), so a
        # small penalty is used instead -- it only wins when genuinely the
        # better match, while a specialist can still win close calls.
        self.default_model_penalty = default_model_penalty

        # Bonuses validated empirically via probe_head_correlation.py against
        # test_prompts_100_4_models.csv. DO NOT raise these above 0.0 without
        # re-running that probe against the current catalog/test set first --
        # these mappings are specific to the *observed* label distributions:
        #   - reasoning: 100% of llama3.2:3b prompts score "No", but only 44%
        #     of math prompts score "Yes". High precision, low recall -> only
        #     bonus the math model when reasoning="Yes".
        #   - domain_knowledge: llama3.2:3b skews "Low", coder is 100% "High",
        #     math is 60% "High". Bonus specialists on "High", bonus the
        #     default on "Low". qwen2.5:3b excluded -- no clean separation on
        #     this head in the probe, and task_type already isolates it well.
        self.reasoning_bonus = reasoning_bonus
        self.domain_bonus = domain_bonus

        self.model_names: list[str] = []
        self.model_task_types: list[set[str]] = []
        self.description_embeddings: np.ndarray | None = None
        self.example_embeddings: np.ndarray | None = None

    # Raw label indices, inverted from get_head_labels() output (name ->
    # index in that dict, so these are index -> meaning). Re-verify against
    # get_head_labels() if the classifier checkpoint ever changes.
    _REASONING_YES = "1"
    _DOMAIN_HIGH = "0"
    _DOMAIN_LOW = "1"

    # Which models get a bonus for which head+label combination -- built
    # from the probe_head_correlation.py run in conversation, not guessed.
    _REASONING_BONUS_MODELS = {"mightykatun/qwen2.5-math:1.5b"}
    _DOMAIN_HIGH_BONUS_MODELS = {"qwen2.5-coder:3b", "mightykatun/qwen2.5-math:1.5b"}
    _DOMAIN_LOW_BONUS_MODELS = {"llama3.2:3b"}

    def load(self, catalog: list[dict]) -> None:
        self.model_names = [entry["model_id"] for entry in catalog]
        self.model_task_types = [set(entry.get("task_types", [])) for entry in catalog]

        descriptions = [entry.get("description", "") for entry in catalog]

        raw_description_embeddings = self.embedder.encode(
            descriptions,
            convert_to_numpy=True,
            show_progress_bar=False,
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

            raw_examples = self.embedder.encode(
                examples,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            example_embeddings[model_idx, : len(examples), :] = _normalize(raw_examples).astype(np.float32)

        self.example_embeddings = example_embeddings

    def _candidate_indices_for_task_type(self, task_type: str) -> list[int]:
        # No default-exclusion: the General Purpose model can be a genuine
        # candidate for a shared task_type (e.g. "Open QA" covers both
        # general chat AND math/logic under NVIDIA's taxonomy). Excluding it
        # outright broke its own legitimate traffic -- see conversation
        # history. Disambiguation happens via default_model_penalty and the
        # head bonuses in _similarity_over_candidates instead.
        return [idx for idx, task_types in enumerate(self.model_task_types) if task_type in task_types]

    def _apply_head_bonuses(
        self,
        routing_score: np.ndarray,
        candidate_indices: list[int],
        classification,
    ) -> np.ndarray:
        """Nudge routing_score using the NVIDIA classifier's auxiliary heads,
        for the specific model/label pairs validated via
        probe_head_correlation.py. Additive bonuses, not hard filters --
        a weak/wrong signal here can only lose a close call, never override
        a strong similarity match."""
        reasoning_head = classification.get_head("reasoning")
        domain_head = classification.get_head("domain_knowledge")

        for local_idx, global_idx in enumerate(candidate_indices):
            model_name = self.model_names[global_idx]

            if (
                self.reasoning_bonus
                and reasoning_head is not None
                and reasoning_head.label == self._REASONING_YES
                and model_name in self._REASONING_BONUS_MODELS
            ):
                routing_score[local_idx] += self.reasoning_bonus

            if self.domain_bonus and domain_head is not None:
                if domain_head.label == self._DOMAIN_HIGH and model_name in self._DOMAIN_HIGH_BONUS_MODELS:
                    routing_score[local_idx] += self.domain_bonus
                elif domain_head.label == self._DOMAIN_LOW and model_name in self._DOMAIN_LOW_BONUS_MODELS:
                    routing_score[local_idx] += self.domain_bonus

        return routing_score

    def _similarity_over_candidates(
        self,
        prompt_embedding: np.ndarray,
        candidate_indices: list[int],
        classification=None,
    ) -> dict:
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

        # Head bonuses first (additive nudge based on classifier signals),
        # then the default-model penalty -- order matters: the penalty
        # should act on the score *after* specialists get their bonus, so a
        # specialist with a genuine head-signal match isn't penalized away
        # by a default that hasn't been nudged down yet.
        if classification is not None:
            routing_score = self._apply_head_bonuses(routing_score, candidate_indices, classification)

        if len(candidate_indices) > 1 and self.general_purpose_model in (
            self.model_names[i] for i in candidate_indices
        ):
            gp_local_idx = next(
                local_idx
                for local_idx, global_idx in enumerate(candidate_indices)
                if self.model_names[global_idx] == self.general_purpose_model
            )
            routing_score[gp_local_idx] -= self.default_model_penalty

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
        """
        Full pipeline, mirroring router.py's route_message(): classify,
        decide ambiguity, filter candidates by task_type, run similarity
        on the candidate pool only, or fall back to General Purpose.

        Returns a dict with the winning model plus enough detail to debug:
        selected_model, is_fallback, task_type, task_type_confidence,
        task_type_margin, and (when not a fallback) the similarity
        breakdown from _similarity_over_candidates.
        """
        classification = classify(prompt)
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

        similarity_result = self._similarity_over_candidates(
            prompt_embedding, candidate_indices, classification=classification
        )

        return {
            **base_result,
            **similarity_result,
            "is_fallback": False,
            "fallback_reason": None,
        }


# =============================================================================
# Test prompt loading
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
# Evaluation loop
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
        "prompt",
        "expected_model",
        "selected_model",
        "correct",
        "is_fallback",
        "fallback_reason",
        "task_type",
        "task_type_confidence",
        "task_type_margin",
        "routing_score",
        "description_sim",
        "example_sim",
        "runner_up_model",
        "similarity_margin",
        "candidate_pool_size",
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
    print(f"ACCURACY: {correct}/{total} ({accuracy:.1%})")
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

    # Confusion matrix: expected_model -> {selected_model: count}
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
            continue  # perfect on this model, skip for brevity

        any_mistakes = True
        print(f"  Expected '{expected_model}' ({correct_for_model}/{total_for_model} correct):")
        for selected_model, count in sorted(row.items(), key=lambda kv: -kv[1]):
            if selected_model == expected_model:
                continue
            print(f"      -> misrouted to '{selected_model}': {count} time(s)")

    if not any_mistakes:
        print("  No mistakes — every model routed 100% correctly on this test set.")

    single_candidate_count = sum(
        1 for r in results if not r["is_fallback"] and r["candidate_pool_size"] == 1
    )
    if single_candidate_count:
        print(
            f"\n[*] {single_candidate_count}/{total} non-fallback prediction(s) had exactly one "
            f"candidate for their task_type — no real competition happened."
        )

    close_calls = sorted(
        (
            r for r in results
            if r["correct"] and not r["is_fallback"] and r["similarity_margin"] is not None
        ),
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
    parser = argparse.ArgumentParser(description="Evaluate the semantic router against labeled test prompts.")
    parser.add_argument("--catalog", default="src/catalog/models.yaml", help="Path to the models YAML catalog.")
    parser.add_argument("--test-prompts", default="test_prompts.csv", help="Path to the labeled test prompts CSV.")
    parser.add_argument("--output", default="results.csv", help="Path to write the results CSV.")
    parser.add_argument(
        "--embedding-model",
        default=os.getenv("EMBEDDING_MODEL_NAME", "sentence-transformers/all-MiniLM-L6-v2"),
        help="Sentence-transformers model name (should match what the live router uses).",
    )
    # Defaults below MUST track router.py's DESCRIPTION_WEIGHT/EXAMPLE_WEIGHT
    # and CLASSIFIER_MARGIN_THRESHOLD constants — if you change those in
    # router.py, change these defaults too, or pass matching flags.
    parser.add_argument("--description-weight", type=float, default=0.4, help="Weight for description similarity.")
    parser.add_argument("--example-weight", type=float, default=0.6, help="Weight for closest-example similarity.")
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=float(os.getenv("CLASSIFIER_MARGIN_THRESHOLD", "0.10")),
        help="Classifier top1/top2 margin below which a prompt is treated as ambiguous.",
    )
    parser.add_argument(
        "--general-purpose-model",
        default=os.getenv("GENERAL_PURPOSE_MODEL"),
        help="Override the General Purpose model id. If omitted, resolved from "
             "the catalog's is_default: true entry, same as router.py.",
    )
    parser.add_argument(
        "--classifier-device",
        default=os.getenv("CLASSIFIER_DEVICE", "cpu"),
        help="cpu or cuda, passed through to the classifier.",
    )
    parser.add_argument(
        "--default-model-penalty",
        type=float,
        default=0.03,
        help="Score penalty applied to the General Purpose model when it "
             "competes against specialists on a shared task_type.",
    )
    parser.add_argument(
        "--reasoning-bonus",
        type=float,
        default=0.0,
        help="Score bonus for the math model when the classifier's "
             "'reasoning' head says Yes. 0.0 disables it (default).",
    )
    parser.add_argument(
        "--domain-bonus",
        type=float,
        default=0.0,
        help="Score bonus for specialists on domain_knowledge=High / for "
             "the default model on domain_knowledge=Low. 0.0 disables it.",
    )
    args = parser.parse_args()

    if not Path(args.catalog).exists():
        sys.exit(f"[!] Catalog file not found: {args.catalog}")
    if not Path(args.test_prompts).exists():
        sys.exit(f"[!] Test prompts file not found: {args.test_prompts}")

    # classifier.py reads CLASSIFIER_DEVICE from the environment at import
    # time, so set it before load_classifier() runs if the user passed
    # --classifier-device explicitly.
    os.environ["CLASSIFIER_DEVICE"] = args.classifier_device

    print(f"[*] Loading catalog from '{args.catalog}'...")
    catalog = load_catalog_from_yaml(args.catalog)

    general_purpose_model = resolve_general_purpose_model(catalog, args.general_purpose_model)
    print(f"[OK] General Purpose model: '{general_purpose_model}'")

    print(f"[*] Loading test prompts from '{args.test_prompts}'...")
    test_rows = load_test_prompts(args.test_prompts)
    print(f"[*] {len(test_rows)} test prompt(s) loaded.")

    print("[*] Loading task classifier...")
    load_classifier()

    print(f"[*] Loading embedding model '{args.embedding_model}' (CPU)...")
    embedder = SentenceTransformer(args.embedding_model, device="cpu")

    engine = RouterEvalEngine(
        embedder=embedder,
        description_weight=args.description_weight,
        example_weight=args.example_weight,
        margin_threshold=args.margin_threshold,
        general_purpose_model=general_purpose_model,
        default_model_penalty=args.default_model_penalty,
        reasoning_bonus=args.reasoning_bonus,
        domain_bonus=args.domain_bonus,
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