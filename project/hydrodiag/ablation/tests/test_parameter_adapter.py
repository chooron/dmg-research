import numpy as np
from ablation.ic_core.parameter_adapter import (
    get_parameter_spec,
    normalized_to_physical,
    parameter_summary,
    physical_to_normalized,
)

MODEL_KEYS = [
    "XAJ",
    "XAJ_CN",
    "XAJ_PD",
    "XAJ_TGD2",
    "GR4J",
    "GR4J_CN",
    "GR4J_PD",
    "GR4J_TGD2",
    "SIMHYD",
    "SIMHYD_CN",
    "SIMHYD_PD",
    "SIMHYD_TGD2",
    "HBV",
]


def test_parameter_dimension_all_models() -> None:
    assert [len(get_parameter_spec(key)) for key in MODEL_KEYS] == [
        15,
        17,
        17,
        17,
        4,
        6,
        6,
        6,
        10,
        12,
        12,
        12,
        12,
    ]


def test_parameter_round_trip_all_models() -> None:
    for key in MODEL_KEYS:
        theta = np.full((2, len(get_parameter_spec(key))), 0.5)
        physical = normalized_to_physical(key, theta)
        recovered = physical_to_normalized(key, physical)
        assert np.allclose(recovered, theta, atol=1e-12), key


def test_tgd2_log_parameter_mapping_and_bounds() -> None:
    physical = normalized_to_physical("XAJ_TGD2", np.full(17, 0.5))
    spec = get_parameter_spec("XAJ_TGD2")
    assert list(spec)[:2] == ["tgd_tau_warm", "tgd_delta_tau_cold"]
    assert np.isclose(
        physical[0],
        np.sqrt(spec["tgd_tau_warm"]["lower"] * spec["tgd_tau_warm"]["upper"]),
    )
    assert np.isclose(
        physical[1],
        np.sqrt(
            spec["tgd_delta_tau_cold"]["lower"] * spec["tgd_delta_tau_cold"]["upper"]
        ),
    )
    assert parameter_summary("XAJ_TGD2")["log_scaled_parameters"] == [
        "tgd_delta_tau_cold",
        "tgd_tau_warm",
    ]


def test_parameter_summary_uses_registry() -> None:
    summary = parameter_summary("XAJ")
    assert summary["parameter_count"] == 15
    assert summary["parameter_names"][0] == "xaj_k"
