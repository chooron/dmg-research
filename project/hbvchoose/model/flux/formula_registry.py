"""Candidate formula registry for HBV Formula-MoE.

Each node has a routing_policy and a list of formula entries.
"""

import copy

FORMULA_REGISTRY = {
    "snow": {
        "routing_policy": "sparse_or_top1",
        "formulas": [
            {
                "id": "S0",
                "name": "hbv_degree_day",
                "function": "snowmelt_linear_degreeday",
                "spec_fid": "snowmelt_linear_degreeday",
                "status": "main",
                "source_level": "HBV default",
            },
            {
                "id": "S4",
                "name": "seasonal_degree_day",
                "function": "cfmax_seasonal + snowmelt_linear_degreeday",
                "spec_fid": "cfmax_seasonal_linear",
                "status": "main",
                "source_level": "HBV snow routine modification",
            },
            {
                "id": "S5",
                "name": "exponential_snowmelt",
                "function": "snowmelt_exponential",
                "spec_fid": "snowmelt_exponential",
                "status": "main",
                "source_level": "HBV snow routine modification",
            },
        ],
    },
    "recharge": {
        "routing_policy": "hard_only",
        "formulas": [
            {
                "id": "R0",
                "name": "hbv_beta_recharge",
                "function": "beta_recharge",
                "spec_fid": "beta_recharge",
                "status": "main",
                "source_level": "HBV default",
            },
            {
                "id": "R4",
                "name": "logistic_saturation_threshold_recharge",
                "function": "saturation_threshold_recharge",
                "spec_fid": "saturation_threshold_recharge",
                "status": "main",
                "source_level": "recommended extension",
            },
            {
                "id": "R5",
                "name": "variable_contributing_area_recharge",
                "function": "variable_contributing_area_recharge",
                "spec_fid": "variable_contributing_area_recharge",
                "status": "main",
                "source_level": "XAJ/VIC-style variable contributing area",
            },
            {
                "id": "R1",
                "name": "linear_recharge_beta_1",
                "function": "linear_recharge",
                "spec_fid": "linear_recharge",
                "status": "ablation_only",
                "source_level": "HBV beta special case",
            },
            {
                "id": "R2",
                "name": "strong_nonlinear_beta_recharge",
                "function": "beta_recharge",
                "spec_fid": "strong_nonlinear_recharge",
                "status": "ablation_only",
                "source_level": "parameter-regime expert",
            },
            {
                "id": "R3",
                "name": "weak_nonlinear_beta_recharge",
                "function": "beta_recharge",
                "spec_fid": "weak_nonlinear_recharge",
                "status": "ablation_only",
                "source_level": "parameter-regime expert",
            },
        ],
    },
    "aet": {
        "routing_policy": "sparse_or_top1",
        "formulas": [
            {
                "id": "E0",
                "name": "hbv_default_aet",
                "function": "aet_hbv_default",
                "spec_fid": "aet_hbv_default",
                "status": "main",
                "source_level": "HBV default",
            },
            {
                "id": "E3",
                "name": "power_law_aet_stress",
                "function": "aet_power_law",
                "spec_fid": "aet_power_law",
                "status": "main",
                "source_level": "recommended extension",
            },
            {
                "id": "E4",
                "name": "feddes_threshold_aet",
                "function": "feddes_threshold_aet",
                "spec_fid": "feddes_threshold_aet",
                "status": "main",
                "source_level": "Feddes-style soil moisture stress",
            },
            {
                "id": "E2",
                "name": "temperature_corrected_pet_hbv_aet",
                "function": "temperature_corrected_aet",
                "spec_fid": "temperature_corrected_aet",
                "status": "pet_correction",
                "source_level": "HBV-light PET correction (not an AET formula competitor)",
            },
        ],
    },
    "response": {
        "routing_policy": "sparse_or_top1",
        "formulas": [
            {
                "id": "Q0",
                "name": "hbv_two_reservoir_response",
                "function": "response_two_reservoir",
                "spec_fid": "response_two_reservoir",
                "status": "main",
                "source_level": "HBV default",
            },
            {
                "id": "Q2",
                "name": "nonlinear_reservoir_response",
                "function": "response_nonlinear",
                "spec_fid": "response_nonlinear",
                "status": "main",
                "source_level": "HBV response variant",
            },
            {
                "id": "Q5",
                "name": "delayed_response",
                "function": "response_delayed_step",
                "spec_fid": "response_delayed_step",
                "status": "extension_only",
                "source_level": "HBV-light delayed response",
                "implementation_status": "partial",
            },
        ],
    },
}


def get_node_formulas(node: str, status: str = "main") -> list[dict]:
    """Return formula entries for *node* matching the given *status*.

    Returns a shallow copy so callers cannot mutate the registry via the
    returned list.  If *node* is unknown a ``ValueError`` is raised; an
    empty list is returned when no entry matches the requested *status*.
    """
    if node not in FORMULA_REGISTRY:
        raise ValueError(f"Unknown registry node: '{node}'.  Known: {list(FORMULA_REGISTRY)}")
    entries = FORMULA_REGISTRY[node]["formulas"]
    return [copy.copy(e) for e in entries if e["status"] == status]


def get_all_main_formulas() -> dict[str, list[dict]]:
    """Return {node: [main-formula-entries, ...]} for all registered nodes."""
    result = {}
    for node in FORMULA_REGISTRY:
        main = get_node_formulas(node, "main")
        if main:
            result[node] = main
    return result


def get_routing_policy(node: str) -> str:
    """Return the MoE routing policy for *node* (e.g. 'hard_only')."""
    if node not in FORMULA_REGISTRY:
        raise ValueError(f"Unknown registry node: '{node}'.  Known: {list(FORMULA_REGISTRY)}")
    return FORMULA_REGISTRY[node]["routing_policy"]


def list_formula_nodes() -> list[str]:
    """Return sorted list of all formula-process nodes in the registry."""
    return sorted(FORMULA_REGISTRY.keys())
