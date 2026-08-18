"""Consistency checks for the canonical model/test registry."""

from models import BaseHydrologicalModel
from tests.model_registry import MODEL_REGISTRY, MODEL_BY_NAME


def test_model_registry_has_unique_complete_entries():
    names = [case.name for case in MODEL_REGISTRY]
    assert len(names) == len(set(names))
    assert set(names) == {
        "HBV",
        "GR4J",
        "XAJ",
        "SIMHYD",
        "CemaNeige",
        "CemaNeigeHyst",
        "PrecipitationDelay",
        "TemperatureDependentGenericDelay2",
        "GR4JWithCemaNeige",
        "GR4JWithPrecipitationDelay",
        "XAJWithCemaNeige",
        "XAJWithPrecipitationDelay",
        "SIMHYDWithCemaNeige",
        "SIMHYDWithPrecipitationDelay",
        "GR4JWithTGD2",
        "SIMHYDWithTGD2",
        "XAJWithTGD2",
    }
    assert set(MODEL_BY_NAME) == set(names)


def test_registered_models_match_parameter_specs_and_interface():
    for case in MODEL_REGISTRY:
        model = case.model_cls()
        assert isinstance(model, BaseHydrologicalModel), case.name
        assert set(model.parameter_specs) == set(case.parameter_specs), case.name
        assert len(model.parameter_specs) == len(case.parameter_specs), case.name
        for name, spec in case.parameter_specs.items():
            assert spec["lower"] <= spec["default"] <= spec["upper"], (
                case.name,
                name,
            )
        if case.has_gamma_uh:
            assert getattr(model, "routing_method", None) == "gamma", case.name
