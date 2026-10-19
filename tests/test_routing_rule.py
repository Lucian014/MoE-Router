"""The deterministic model-selection rule (src/router/routing_rule.py)."""
from pathlib import Path

import pytest

from router.classifier import CATEGORIES
from router.routing_rule import Model, catalog_from_rows, load_catalog, select_model, validate_catalog

CATALOG_YAML = Path(__file__).resolve().parents[1] / "src" / "catalog" / "models.yaml"


def catalog(*models):
    return list(models)


DEFAULT = Model("general:4b", category="open-qa", complexity_tier="high", is_default=True)


# ------------------------------------------------------------------ select_model
def test_exact_match_on_category_and_tier():
    cat = catalog(DEFAULT,
                  Model("coder:1b", category="coding", complexity_tier="low"),
                  Model("coder:7b", category="coding", complexity_tier="high"))
    assert select_model(cat, "coding", "low") == ("coder:1b", "match")
    assert select_model(cat, "coding", "high") == ("coder:7b", "match")


def test_missing_tier_falls_back_to_other_tier():
    cat = catalog(DEFAULT, Model("coder:1b", category="coding", complexity_tier="low"))
    assert select_model(cat, "coding", "high") == ("coder:1b", "other-level")


def test_no_model_in_category_falls_back_to_default():
    cat = catalog(DEFAULT, Model("coder:1b", category="coding", complexity_tier="low"))
    assert select_model(cat, "math", "high") == ("general:4b", "fallback:no-model")


def test_disabled_models_are_never_selected():
    cat = catalog(DEFAULT,
                  Model("coder:1b", category="coding", complexity_tier="low"),
                  Model("coder:7b", category="coding", complexity_tier="high", enabled=False))
    assert select_model(cat, "coding", "high") == ("coder:1b", "other-level")


def test_disabled_category_falls_back_to_default():
    cat = catalog(DEFAULT, Model("coder:1b", category="coding", complexity_tier="low", enabled=False))
    assert select_model(cat, "coding", "low") == ("general:4b", "fallback:no-model")


def test_confidence_threshold_is_off_by_default():
    cat = catalog(DEFAULT, Model("coder:1b", category="coding", complexity_tier="low"))
    assert select_model(cat, "coding", "low", confidence=0.01) == ("coder:1b", "match")
    assert select_model(cat, "coding", "low", confidence=0.01, conf_threshold=0.5) == \
        ("general:4b", "fallback:low-confidence")


@pytest.mark.parametrize("defaults", [0, 2])
def test_select_requires_exactly_one_enabled_default(defaults):
    cat = [Model(f"d{i}", category="open-qa", complexity_tier="low", is_default=True) for i in range(defaults)]
    with pytest.raises(ValueError, match="exactly one enabled is_default"):
        select_model(cat, "open-qa", "low")


def test_disabled_default_does_not_count():
    cat = catalog(Model("old", category="open-qa", complexity_tier="low", is_default=True, enabled=False), DEFAULT)
    assert select_model(cat, "math", "low") == ("general:4b", "fallback:no-model")


# ------------------------------------------------------------------ validate_catalog
def test_valid_catalog_has_no_errors():
    cat = catalog(DEFAULT,
                  Model("coder:1b", category="coding", complexity_tier="low"),
                  Model("coder:7b", category="coding", complexity_tier="high"))
    errors, _ = validate_catalog(cat, CATEGORIES)
    assert errors == []


def test_two_enabled_models_on_same_category_and_tier_is_an_error():
    cat = catalog(DEFAULT,
                  Model("a", category="coding", complexity_tier="low"),
                  Model("b", category="coding", complexity_tier="low"))
    errors, _ = validate_catalog(cat, CATEGORIES)
    assert any("two enabled low models" in e for e in errors)


def test_same_slot_is_fine_when_one_is_disabled():
    cat = catalog(DEFAULT,
                  Model("a", category="coding", complexity_tier="low"),
                  Model("b", category="coding", complexity_tier="low", enabled=False))
    errors, _ = validate_catalog(cat, CATEGORIES)
    assert errors == []


@pytest.mark.parametrize("model, message", [
    (Model("x", category="cooking", complexity_tier="low"), "is not a domain label"),
    (Model("x", category="coding", complexity_tier="medium"), "complexity_tier must be one of"),
])
def test_invalid_category_or_tier_is_an_error(model, message):
    errors, _ = validate_catalog(catalog(DEFAULT, model), CATEGORIES)
    assert any(message in e for e in errors)


def test_duplicate_model_id_is_an_error():
    cat = catalog(DEFAULT, Model("general:4b", category="coding", complexity_tier="low"))
    errors, _ = validate_catalog(cat, CATEGORIES)
    assert any("duplicate model_id" in e for e in errors)


def test_warnings_for_uncovered_categories_and_single_tier():
    cat = catalog(DEFAULT, Model("coder:1b", category="coding", complexity_tier="low"))
    _, warnings = validate_catalog(cat, CATEGORIES)
    assert any("'math' has no enabled model" in w for w in warnings)
    assert any("'coding' has only a low model" in w for w in warnings)


# ------------------------------------------------------------------ catalog_from_rows
def test_rows_with_legacy_task_types_list():
    [m] = catalog_from_rows([{"model_id": "a", "task_types": ["coding"], "complexity_tier": "low"}])
    assert m.category == "coding"


def test_rows_with_several_task_types_are_rejected():
    with pytest.raises(ValueError, match="exactly one category"):
        catalog_from_rows([{"model_id": "a", "task_types": ["coding", "math"], "complexity_tier": "low"}])


def test_row_defaults():
    [m] = catalog_from_rows([{"model_id": "a", "category": "coding", "complexity_tier": "low"}])
    assert (m.enabled, m.is_default, m.engine) == (True, False, "ollama")


# ------------------------------------------------------------------ the real catalog
def test_shipped_catalog_is_valid():
    errors, _ = validate_catalog(load_catalog(CATALOG_YAML), CATEGORIES)
    assert errors == []
