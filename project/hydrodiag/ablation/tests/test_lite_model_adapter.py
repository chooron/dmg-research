from ablation.ic_core.model_adapter import (
    LITE_MODEL_CLASSES,
    MODEL_CLASSES,
    model_variant_inventory,
)


def test_all_ic_models_have_explicit_lite_mapping() -> None:
    assert set(LITE_MODEL_CLASSES) == set(MODEL_CLASSES)
    assert len(model_variant_inventory()) == len(MODEL_CLASSES)
    assert all(row["native_lite"] for row in model_variant_inventory())


def test_lite_class_names_are_distinct_from_full_classes() -> None:
    for row in model_variant_inventory():
        assert row["lite_class"].endswith("Lite")
        assert row["lite_class"] != row["full_class"]
