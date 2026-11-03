#!/usr/bin/env python3
"""
evaluate_router_laya.py

End-to-end evaluation of the fine-tuned Laya decision model (v1: ONE checkpoint that
answers both questions, task type + complexity) plugged into the router's selection rule.
Same harness shape as the old evaluate_router*.py scripts: labelled CSV in
(prompt,expected_model), results CSV out, summary + confusion matrices on stdout.

Selection rule evaluated (CLAUDE.md, section 1):
  1. margin(domain) below threshold              -> default model (is_default: true)
  2. candidates = models whose task_types contain the predicted category
  3. candidate whose complexity_tier == predicted level
  4. no candidate at that level                  -> nearest level (ties -> lower level)
  5. no candidate at all                         -> default model
  If candidates in a category have NO complexity_tier in models.yaml yet, the rule can't
  choose between them: the script takes the default model if it is a candidate, otherwise
  the first one, and counts the prompt under "no_tier_in_catalog" so it's visible.
  --tiers lets you try a tier assignment without editing models.yaml, e.g.
      --tiers "codegemma:2b=low,qwen2.5-coder:3b=high,mightykatun/qwen2.5-math:1.5b=low,llama3.2:3b=high"

Reports
  A. Domain classifier   accuracy, per-category precision / recall / F1, macro-F1, confusion matrix
                         (expected category = task_types of expected_model in the catalog)
  B. Routing             expected_model accuracy, model confusion, fallback counts
  C. Complexity          predicted level distribution per expected model (no gold labels in the CSV)
  D. Threshold sweep     accuracy vs coverage of the domain classifier, to pick the fallback threshold
  E. Latency             per-prompt classification time
  Every prediction with ALL probabilities goes to --output (CLAUDE.md: log everything for calibration).

Usage (from the repo root):
    python scripts/evaluate_router_laya.py \
        --checkpoint src/router/laya-router/checkpoints/laya-router \
        --schema src/router/laya-router/laya_task_schema.py \
        --catalog src/catalog/models.yaml \
        --test-prompts scripts/resources/test_prompts_complex.csv \
        --output results_laya.csv
    # or straight from the Hub (needs HF_TOKEN for a private repo):
    python scripts/evaluate_router_laya.py --checkpoint Lucian04/laya-router ...
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import yaml

TIERS = ["low", "medium", "high"]


# ============================================================================= loading
def load_schema(path: str):
    """Import laya_task_schema.py from a file path: the questions MUST be the exact ones the
    checkpoint was trained with, so we never re-declare them here."""
    p = Path(path)
    if not p.exists():
        sys.exit(f"[!] Schema file not found: {p}. Pass --schema pointing at the laya_task_schema.py "
                 f"used for training this checkpoint.")
    spec = importlib.util.spec_from_file_location("laya_task_schema", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_catalog(path: str, tier_override: str | None):
    with open(path, encoding="utf-8") as f:
        models = yaml.safe_load(f)["models"]
    overrides = {}
    if tier_override:
        for pair in tier_override.split(","):
            mid, tier = pair.rsplit("=", 1)
            if tier.strip() not in TIERS:
                sys.exit(f"[!] --tiers: '{tier}' is not one of {TIERS}")
            overrides[mid.strip()] = tier.strip()
    catalog = []
    for m in models:
        if not m.get("enabled", True):
            continue
        catalog.append({
            "model_id": m["model_id"],
            "task_types": list(m.get("task_types", [])),
            "complexity_tier": overrides.get(m["model_id"], m.get("complexity_tier")),
            "is_default": bool(m.get("is_default", False)),
        })
    unknown = set(overrides) - {m["model_id"] for m in catalog}
    if unknown:
        sys.exit(f"[!] --tiers names models not in the catalog: {sorted(unknown)}")
    return catalog


def resolve_default(catalog, override: str | None) -> str:
    if override:
        if override not in {m["model_id"] for m in catalog}:
            sys.exit(f"[!] --general-purpose-model '{override}' is not in the catalog.")
        return override
    defaults = [m["model_id"] for m in catalog if m["is_default"]]
    if len(defaults) != 1:
        sys.exit(f"[!] Expected exactly one is_default model, found {defaults}.")
    return defaults[0]


def load_test_prompts(path: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or not {"prompt", "expected_model"} <= set(reader.fieldnames):
            sys.exit(f"[!] {path} needs a header with 'prompt' and 'expected_model'. Found: {reader.fieldnames}")
        rows = [r for r in reader if r.get("prompt", "").strip()]
    if not rows:
        sys.exit(f"[!] No prompts in {path}")
    return rows


# ============================================================================= routing rule
def nearest_tier_candidate(cands, level: str):
    """Exact tier if present, else nearest tier; on a tie the LOWER tier wins (cheaper model).
    Tie-break is an open decision in CLAUDE.md -- change it here when decided."""
    tiered = [m for m in cands if m["complexity_tier"] in TIERS]
    if not tiered:
        return None, "no_tier_in_catalog"
    target = TIERS.index(level)
    best = min(tiered, key=lambda m: (abs(TIERS.index(m["complexity_tier"]) - target),
                                      TIERS.index(m["complexity_tier"])))
    how = "exact_tier" if best["complexity_tier"] == level else "nearest_tier"
    return best, how


def route(domain: str, margin: float, level: str, catalog, default_model: str, threshold: float):
    if margin < threshold:
        return default_model, "fallback_low_margin"
    cands = [m for m in catalog if domain in m["task_types"]]
    if not cands:
        return default_model, "fallback_no_candidate"
    if len(cands) == 1:
        return cands[0]["model_id"], "single_candidate"
    chosen, how = nearest_tier_candidate(cands, level)
    if chosen is None:  # category has several models but none declares a tier yet
        d = [m for m in cands if m["model_id"] == default_model]
        return (d[0] if d else cands[0])["model_id"], how
    return chosen["model_id"], how


# ============================================================================= metrics
def per_class_report(gold, pred, labels):
    rows, f1s = [], []
    for l in labels:
        tp = sum(g == l and p == l for g, p in zip(gold, pred))
        fp = sum(g != l and p == l for g, p in zip(gold, pred))
        fn = sum(g == l and p != l for g, p in zip(gold, pred))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        support = tp + fn
        rows.append((l, prec, rec, f1, support))
        if support:
            f1s.append(f1)
    return rows, (sum(f1s) / len(f1s) if f1s else 0.0)


def print_confusion(gold, pred, labels, title):
    cm = defaultdict(Counter)
    for g, p in zip(gold, pred):
        cm[g][p] += 1
    w = max(len(l) for l in labels)
    short = [l[:7] for l in labels]
    print(f"\n{title}  (rows = expected, columns = predicted)")
    print(" " * (w + 2) + " ".join(f"{s:>7}" for s in short))
    for g in labels:
        if sum(cm[g].values()):
            print(f"{g:>{w}}  " + " ".join(f"{cm[g][p]:7d}" for p in labels))


# ============================================================================= main
def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the fine-tuned Laya router (v1, combined checkpoint).")
    ap.add_argument("--checkpoint", default="src/router/laya-router/checkpoints/laya-router",
                    help="Local folder or HF repo id of the fine-tuned Laya model.")
    ap.add_argument("--schema", default="src/router/laya-router/laya_task_schema.py",
                    help="The laya_task_schema.py the checkpoint was trained with.")
    ap.add_argument("--catalog", default="src/catalog/models.yaml")
    ap.add_argument("--test-prompts", default="scripts/resources/test_prompts_complex.csv")
    ap.add_argument("--output", default="results_laya.csv")
    ap.add_argument("--margin-threshold", type=float, default=0.10,
                    help="top1 - top2 domain probability below this -> default model.")
    ap.add_argument("--tiers", default=None,
                    help='Try complexity tiers without editing models.yaml: "model=low,model2=high"')
    ap.add_argument("--general-purpose-model", default=None)
    ap.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")
    args = ap.parse_args()

    for p in (args.catalog, args.test_prompts):
        if not Path(p).exists():
            sys.exit(f"[!] Not found: {p}")

    schema = load_schema(args.schema)
    questions = schema.QUESTIONS
    task_qid, cx_qid = schema.TASK_QID, schema.COMPLEXITY_QID
    labels = list(schema.LABELS)

    catalog = load_catalog(args.catalog, args.tiers)
    default_model = resolve_default(catalog, args.general_purpose_model)
    model_to_cat = {m["model_id"]: m["task_types"][0] for m in catalog if m["task_types"]}
    rows = load_test_prompts(args.test_prompts)

    unknown_cats = {c for m in catalog for c in m["task_types"]} - set(labels)
    if unknown_cats:
        print(f"[!] Catalog categories the checkpoint was never trained on: {sorted(unknown_cats)}")

    print(f"[*] Catalog: {len(catalog)} models, default='{default_model}'")
    for cat in labels:
        ms = [f"{m['model_id']}({m['complexity_tier'] or '-'})" for m in catalog if cat in m["task_types"]]
        print(f"      {cat:17s} {', '.join(ms) or '(no model)'}")
    print(f"[*] Loading Laya checkpoint '{args.checkpoint}'...")
    import laya  # imported late so --help works without the package
    agent = laya.Agent(args.checkpoint, device=args.device)
    print(f"[*] Evaluating {len(rows)} prompts...")

    results, latencies = [], []
    for i, r in enumerate(rows, 1):
        prompt, expected_model = r["prompt"], r["expected_model"]
        t0 = time.perf_counter()
        ans = agent.predict(prompt, questions)["answers"]
        latencies.append((time.perf_counter() - t0) * 1000)

        dom = ans[task_qid]
        ranked = sorted(dom["probabilities"].items(), key=lambda kv: kv[1], reverse=True)
        (d1, p1), (d2, p2) = ranked[0], ranked[1]
        cx = ans[cx_qid]
        level = TIERS[min(len(TIERS) - 1, int(round(cx["score"])))]

        selected, how = route(d1, p1 - p2, level, catalog, default_model, args.margin_threshold)
        exp_cat = model_to_cat.get(expected_model, "?")
        results.append({
            "prompt": prompt,
            "expected_model": expected_model,
            "selected_model": selected,
            "model_correct": selected == expected_model,
            "expected_category": exp_cat,
            "pred_category": d1,
            "category_correct": d1 == exp_cat,
            "category_prob": round(p1, 4),
            "runner_up": d2,
            "runner_up_prob": round(p2, 4),
            "margin": round(p1 - p2, 4),
            "pred_complexity": level,
            "complexity_score": round(cx["score"], 4),
            "route": how,
            "latency_ms": round(latencies[-1], 1),
            **{f"p_{k}": v for k, v in dom["probabilities"].items()},
            **{f"p_cx_{TIERS[int(k)]}": v for k, v in cx["probabilities"].items()},
        })
        if i % 25 == 0:
            print(f"    {i}/{len(rows)}")

    with open(args.output, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)

    n = len(results)
    # ---------------------------------------------------------------- A. domain
    scored = [x for x in results if x["expected_category"] != "?"]
    gold = [x["expected_category"] for x in scored]
    pred = [x["pred_category"] for x in scored]
    acc = sum(g == p for g, p in zip(gold, pred)) / max(1, len(scored))
    report, macro_f1 = per_class_report(gold, pred, labels)
    print("\n" + "=" * 72)
    print(f"A. DOMAIN CLASSIFIER   accuracy={acc:.3f}   macro-F1={macro_f1:.3f}   (n={len(scored)})")
    print("=" * 72)
    print(f"{'category':17s} {'precision':>9} {'recall':>7} {'F1':>6} {'n':>4}")
    for l, pr, rc, f1, sup in report:
        if sup or pr:
            print(f"{l:17s} {pr:9.2f} {rc:7.2f} {f1:6.2f} {sup:4d}")
    print_confusion(gold, pred, labels, "Domain confusion matrix")
    skipped = n - len(scored)
    if skipped:
        print(f"[!] {skipped} prompts skipped in A: expected_model not in the catalog.")

    # ---------------------------------------------------------------- B. routing
    m_acc = sum(x["model_correct"] for x in results) / n
    print("\n" + "=" * 72)
    print(f"B. END-TO-END ROUTING  expected_model accuracy={m_acc:.3f}   (threshold={args.margin_threshold})")
    print("=" * 72)
    for how, c in Counter(x["route"] for x in results).most_common():
        ok = sum(x["model_correct"] for x in results if x["route"] == how)
        print(f"  {how:22s} {c:4d} prompts   correct {ok:4d}")
    model_ids = sorted({x["expected_model"] for x in results} | {x["selected_model"] for x in results})
    print_confusion([x["expected_model"] for x in results], [x["selected_model"] for x in results],
                    model_ids, "Model confusion")
    # wrong model although the category was right = the within-category choice failed
    same_cat_wrong = sum(x["category_correct"] and not x["model_correct"] for x in results)
    print(f"\n  category right but model wrong: {same_cat_wrong}  "
          f"(within-category choice; depends on complexity_tier in the catalog)")

    # ---------------------------------------------------------------- C. complexity
    print("\n" + "=" * 72)
    print("C. COMPLEXITY  predicted level per expected model (CSV has no gold complexity)")
    print("=" * 72)
    by_model = defaultdict(list)
    for x in results:
        by_model[x["expected_model"]].append(x)
    print(f"{'expected_model':32s} {'low':>5} {'med':>5} {'high':>5} {'mean score':>11}")
    for mid in sorted(by_model):
        xs = by_model[mid]
        c = Counter(x["pred_complexity"] for x in xs)
        mean = statistics.mean(x["complexity_score"] for x in xs)
        print(f"{mid:32s} {c['low']:5d} {c['medium']:5d} {c['high']:5d} {mean:11.2f}")
    overall = Counter(x["pred_complexity"] for x in results)
    print(f"{'ALL':32s} {overall['low']:5d} {overall['medium']:5d} {overall['high']:5d}")

    # ---------------------------------------------------------------- D. threshold sweep
    print("\n" + "=" * 72)
    print("D. THRESHOLD SWEEP  (domain decision kept only if margin >= threshold)")
    print("=" * 72)
    print(f"{'threshold':>9} {'coverage':>9} {'accuracy on kept':>17} {'wrong & kept':>13}")
    for t in (0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.85, 0.9, 0.92):
        kept = [x for x in scored if x["margin"] >= t]
        if not kept:
            continue
        a = sum(x["category_correct"] for x in kept) / len(kept)
        wrong = sum(not x["category_correct"] for x in kept)
        print(f"{t:9.2f} {len(kept) / len(scored):9.2f} {a:17.3f} {wrong:13d}")
    wrong_margins = sorted(x["margin"] for x in scored if not x["category_correct"])
    if wrong_margins:
        print(f"  margins of the wrong predictions: min={wrong_margins[0]:.2f} "
              f"median={statistics.median(wrong_margins):.2f} max={wrong_margins[-1]:.2f}")

    # ---------------------------------------------------------------- E. latency
    lat = sorted(latencies)
    print("\n" + "=" * 72)
    print(f"E. LATENCY  mean={statistics.mean(lat):.0f} ms  p50={lat[len(lat) // 2]:.0f} ms  "
          f"p95={lat[int(len(lat) * 0.95) - 1]:.0f} ms  (first call includes warm-up)")
    print("=" * 72)
    print(f"\n[OK] {n} predictions with all probabilities -> {args.output}")


if __name__ == "__main__":
    main()