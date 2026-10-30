"""
routing_rule.py -- the router's deterministic model-selection rule

Every model is an expert for ONE category and ONE complexity tier (low / high).
Rule:
  1. candidates = ENABLED models whose category == the predicted category (0, 1 or 2 models)
  2. the candidate with complexity_tier == predicted level          -> "match"
  3. that level missing in the category -> the other level           -> "other-level"
  4. no enabled model in the category   -> the is_default model      -> "fallback:no-model"

There is no fallback on classifier confidence (decided 9 Oct, see docs/laya_training_log.md):
`conf_threshold` exists only for experiments and defaults to 0 (never triggers).
"""
from dataclasses import dataclass

LEVELS = ["low", "high"]


@dataclass
class Model:
    model_id: str
    category: str | None = None
    complexity_tier: str | None = None
    is_default: bool = False
    enabled: bool = True
    engine: str = "ollama"
    # the model's token budget (models.context_window / max_output_tokens), for context assembly
    context_window: int = 4096
    max_output_tokens: int = 512


def find_model(catalog, model_id):
    return next(m for m in catalog if m.model_id == model_id)


def _category_of(row):
    if "category" in row:
        return row["category"] or None
    task_types = list(row.get("task_types") or [])
    if len(task_types) > 1:
        raise ValueError(f"{row['model_id']}: a model serves exactly one category, "
                         f"got task_types={task_types}")
    return task_types[0] if task_types else None


def catalog_from_rows(rows):
    return [Model(model_id=r["model_id"], category=_category_of(r),
                  complexity_tier=r.get("complexity_tier"), is_default=bool(r.get("is_default", False)),
                  enabled=bool(r.get("enabled", True)), engine=r.get("engine", "ollama"),
                  context_window=int(r.get("context_window") or 4096),
                  max_output_tokens=int(r.get("max_output_tokens") or 512))
            for r in rows]


def load_catalog(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return catalog_from_rows(yaml.safe_load(f)["models"])


def default_model(catalog):
    defaults = [m for m in catalog if m.is_default and m.enabled]
    if len(defaults) != 1:
        raise ValueError(f"catalog must have exactly one enabled is_default model, found {len(defaults)}")
    return defaults[0]


def select_model(catalog, category, level, confidence=1.0, conf_threshold=0.0):
    """-> (model_id, reason)"""
    default = default_model(catalog)
    if confidence < conf_threshold:
        return default.model_id, "fallback:low-confidence"
    candidates = [m for m in catalog if m.enabled and m.category == category]
    if not candidates:
        return default.model_id, "fallback:no-model"
    for m in candidates:
        if m.complexity_tier == level:
            return m.model_id, "match"
    return candidates[0].model_id, "other-level"


def validate_catalog(catalog, categories):
    """-> (errors, warnings); categories = the domain classifier's labels."""
    errors, warnings = [], []
    try:
        default_model(catalog)
    except ValueError as e:
        errors.append(str(e))
    ids = [m.model_id for m in catalog]
    for mid in {i for i in ids if ids.count(i) > 1}:
        errors.append(f"duplicate model_id {mid}")
    seen = {}
    for m in catalog:
        if m.complexity_tier not in LEVELS:
            errors.append(f"{m.model_id}: complexity_tier must be one of {LEVELS}, got {m.complexity_tier!r}")
        if m.category is None:
            if not m.is_default:
                warnings.append(f"{m.model_id}: no category and not the default -> never selected")
            continue
        if m.category not in categories:
            errors.append(f"{m.model_id}: category {m.category!r} is not a domain label {categories}")
        if m.enabled:
            key = (m.category, m.complexity_tier)
            if key in seen:
                errors.append(f"category {m.category!r} has two enabled {m.complexity_tier} models: "
                              f"{seen[key]} and {m.model_id}")
            seen[key] = m.model_id
    for c in categories:
        tiers = {t for (cat, t) in seen if cat == c}
        if not tiers:
            warnings.append(f"category {c!r} has no enabled model -> always the default model")
        elif len(tiers) == 1:
            warnings.append(f"category {c!r} has only a {next(iter(tiers))} model -> used for both levels")
    return errors, warnings