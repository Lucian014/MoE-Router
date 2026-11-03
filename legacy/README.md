# legacy/

Files the project no longer uses, kept for reference (the thesis cites earlier stages of the
router). Same layout as the repository root. Nothing here is imported, built or tested:
ruff and the Docker builds skip this folder.

| file | what it was | replaced by |
|---|---|---|
| `scripts/evaluate_router.py` | the first router: NVIDIA DeBERTa task classifier + embedding similarity over the catalog | the Laya classifiers (`src/router/classifier.py`) + `routing_rule.py` |
| `scripts/evaluate_router_logistic_regression.py` | the same harness with a logistic-regression classifier | idem |
| `scripts/evaluate_router_laya.py` | Laya v1: ONE checkpoint answering task type + complexity | two independent checkpoints (laya-domain-v4, laya-complexity-v2), evaluated with `scripts/laya/eval_laya.py` |
| `scripts/laya/laya_classifier.py` | the router-side wrapper of Laya v1 | `src/router/classifier.py` |
| `scripts/fix_domain_datasets.py` | (empty) | the fix rules in `scripts/laya/build_domain_dataset.py` |
| `scripts/convert_json_to_jsonl.py` | converts the HF prompt datasets (.json arrays) to .jsonl | — (run once; the datasets are .jsonl now) |
| `scripts/label_prompt_categories.py` | labels prompts with an LLM (tool calling) into the categories of `categories.yaml` | — (run once to build the domain datasets) |
| `src/catalog/categories.yaml` | the category catalog of the embedding router (descriptions + examples) | the labels in `src/router/domain_schema.py` |
