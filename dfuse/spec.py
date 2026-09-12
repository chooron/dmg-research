"""Single authority for the extracted FUSE-78 structural specification.

The catalogue data are generated from the paper repository's
``list_decision_78.txt``.  This module is deliberately independent of the
upstream Fortran implementation: it only encodes the public decision/state/
flux contract needed by the tensor kernel and experiment layer.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

SPEC_DIR = Path(__file__).with_name("specs")

DECISION_ORDER = (
    "RFERR",
    "ARCH1",
    "ARCH2",
    "QSURF",
    "QPERC",
    "ESOIL",
    "QINTF",
    "Q_TDH",
    "SNOWM",
)

# Integer codes are the public names in model_defnames.f90.  The four open
# structural decisions are kept explicit below; the other values are fixed
# by the paper's 78-row execution list.
DECISION_CODES = {
    "additive_e": 1001,
    "multiplc_e": 1002,
    "tension1_1": 2001,
    "tension2_1": 2002,
    "onestate_1": 2003,
    "tens2pll_2": 3001,
    "unlimfrc_2": 3002,
    "unlimpow_2": 3003,
    "fixedsiz_2": 3004,
    "topmdexp_2": 3005,
    "arno_x_vic": 4001,
    "prms_varnt": 4002,
    "tmdl_param": 4003,
    "perc_f2sat": 5001,
    "perc_w2sat": 5002,
    "perc_lower": 5003,
    "sequential": 6001,
    "rootweight": 6002,
    "intflwnone": 7001,
    "intflwsome": 7002,
    "rout_gamma": 8001,
    "no_routing": 8002,
    "no_snowmod": 8501,
    "temp_index": 8502,
}

OPEN_DECISIONS = ("ARCH1", "ARCH2", "QPERC", "QSURF")
FIXED_PAPER_DECISIONS = {
    "RFERR": "multiplc_e",
    "ESOIL": "sequential",
    "QINTF": "intflwnone",
    "Q_TDH": "rout_gamma",
    "SNOWM": "temp_index",
}

# This is the state super-set used by all 78 models.  FREE_2 is a derived
# diagnostic in the upstream code, not an independent CSTATE entry.
STATE_NAMES = (
    "TENS_1A",
    "TENS_1B",
    "TENS_1",
    "FREE_1",
    "WATR_1",
    "TENS_2",
    "FREE_2A",
    "FREE_2B",
    "WATR_2",
)

PARAMETER_NAMES = tuple(
    json.loads((SPEC_DIR / "parameter_catalog.json").read_text())["parameters"][i]["name"]
    for i in range(37)
)

FLUX_NAMES = (
    "EFF_PPT",
    "SATAREA",
    "EVAP_1A",
    "EVAP_1B",
    "EVAP_1",
    "RCHR2EXCS",
    "TENS2FREE_1",
    "QPERC_12",
    "QINTF_1",
    "OFLOW_1",
    "QSURF",
    "EVAP_2",
    "TENS2FREE_2",
    "QBASE_2A",
    "QBASE_2B",
    "QBASE_2",
    "OFLOW_2A",
    "OFLOW_2B",
    "OFLOW_2",
)

SOLVER_CONFIG = {
    "solution_method": 2,
    "solution_method_name": "implicit_euler",
    "temporal_error_control": 0,
    "temporal_error_control_name": "fixed_time_steps",
    "initial_newton": 0,
    "jacobian_re_evaluation": 0,
    "newton_error_trap": 1,
    "step_end_processing": 1,
    "err_trunc_abs_mm": 1.0e-2,
    "err_trunc_rel": 1.0e-2,
    "err_iter_func": 1.0e-12,
    "err_iter_dx": 1.0e-12,
    "threshold_freeze_jacobian": 1.0e-9,
    "frac_state_min": 1.0e-8,
    "safety_factor": 0.9,
    "min_step_multiplier": 0.1,
    "max_step_multiplier": 4.0,
    "max_iterations": 1000,
    "min_step_minutes": 0.01,
    "max_step_minutes": 1440.0,
}

_ARCH1_STATES = {
    "tension2_1": ("TENS_1A", "TENS_1B", "FREE_1"),
    "tension1_1": ("TENS_1", "FREE_1"),
    "onestate_1": ("WATR_1",),
}
_ARCH2_STATES = {
    "tens2pll_2": ("TENS_2", "FREE_2A", "FREE_2B"),
    "unlimfrc_2": ("WATR_2",),
    "unlimpow_2": ("WATR_2",),
    "fixedsiz_2": ("WATR_2",),
    "topmdexp_2": ("WATR_2",),
}

_ARCH1_FLUXES = {
    "tension2_1": (
        "EFF_PPT", "EVAP_1A", "EVAP_1B", "RCHR2EXCS", "TENS2FREE_1",
        "QPERC_12", "QINTF_1", "OFLOW_1", "QSURF",
    ),
    "tension1_1": (
        "EFF_PPT", "EVAP_1", "TENS2FREE_1", "QPERC_12", "QINTF_1",
        "OFLOW_1", "QSURF",
    ),
    "onestate_1": (
        "EFF_PPT", "EVAP_1", "QPERC_12", "QINTF_1", "OFLOW_1", "QSURF",
    ),
}
_ARCH2_FLUXES = {
    "tens2pll_2": (
        "EVAP_2", "TENS2FREE_2", "QBASE_2A", "QBASE_2B", "QBASE_2",
        "OFLOW_2A", "OFLOW_2B", "OFLOW_2",
    ),
    "unlimfrc_2": ("EVAP_2", "QBASE_2", "OFLOW_2"),
    "unlimpow_2": ("EVAP_2", "QBASE_2", "OFLOW_2"),
    "fixedsiz_2": ("EVAP_2", "QBASE_2", "OFLOW_2"),
    "topmdexp_2": ("EVAP_2", "QBASE_2", "OFLOW_2"),
}


def _load_rows() -> list[dict[str, str]]:
    payload = json.loads((SPEC_DIR / "structures_78.json").read_text())
    return payload["rows"]


def _load_parameters() -> dict[str, dict[str, Any]]:
    payload = json.loads((SPEC_DIR / "parameter_catalog.json").read_text())
    return {item["name"]: item for item in payload["parameters"]}


PARAMETERS = _load_parameters()


def _legal_reason(decisions: Mapping[str, str]) -> str | None:
    if decisions["ARCH1"] == "tension2_1" and decisions["ARCH2"] == "tens2pll_2":
        return "tension2_1 cannot be paired with tens2pll_2"
    if decisions["ARCH1"] != "onestate_1" and decisions["QPERC"] == "perc_w2sat":
        return "perc_w2sat is only legal with onestate_1"
    return None


def _parameter_names(decisions: Mapping[str, str]) -> tuple[str, ...]:
    names: list[str] = ["RFERR_MLT"]
    if decisions["ARCH1"] == "tension2_1":
        names += ["FRCHZNE", "FRACTEN", "MAXWATR_1", "FRACLOWZ"]
    else:
        names += ["FRACTEN", "MAXWATR_1"]

    arch2 = decisions["ARCH2"]
    if arch2 == "tens2pll_2":
        names += ["PERCFRAC", "FPRIMQB", "MAXWATR_2", "QBRATE_2A", "QBRATE_2B"]
    elif arch2 == "unlimfrc_2":
        names += ["MAXWATR_2", "QB_PRMS"]
    elif arch2 == "unlimpow_2":
        names += ["MAXWATR_2", "BASERTE", "LOGLAMB", "TISHAPE", "QB_POWR"]
    elif arch2 == "fixedsiz_2":
        names += ["MAXWATR_2", "BASERTE", "QB_POWR"]
    else:
        raise ValueError(f"unsupported ARCH2 in paper catalogue: {arch2}")

    if decisions["QPERC"] in ("perc_f2sat", "perc_w2sat"):
        names += ["PERCRTE", "PERCEXP"]
    else:
        names += ["SACPMLT", "SACPEXP"]

    if decisions["QSURF"] == "arno_x_vic":
        names += ["AXV_BEXP"]
    elif decisions["QSURF"] == "prms_varnt":
        names += ["SAREAMAX"]
    elif decisions["QSURF"] == "tmdl_param":
        if arch2 in ("tens2pll_2", "unlimfrc_2", "fixedsiz_2"):
            names += ["LOGLAMB", "TISHAPE"]
        if arch2 in ("tens2pll_2", "unlimfrc_2", "topmdexp_2"):
            names += ["QB_POWR"]
    else:
        raise ValueError(f"unsupported QSURF in paper catalogue: {decisions['QSURF']}")

    names += ["TIMEDELAY", "MBASE", "MFMAX", "MFMIN", "PXTEMP", "OPG", "LAPSE"]
    # A repeated parameter is one coordinate in the union, matching LPARAM's
    # named-parameter representation.
    return tuple(dict.fromkeys(names))


def _topology(decisions: Mapping[str, str]) -> dict[str, Any]:
    if decisions["ARCH1"] == "tension2_1":
        upper = {
            "TENS_1A": {"EFF_PPT": 1, "QSURF": -1, "EVAP_1A": -1, "RCHR2EXCS": -1},
            "TENS_1B": {"RCHR2EXCS": 1, "EVAP_1B": -1, "TENS2FREE_1": -1},
            "FREE_1": {"TENS2FREE_1": 1, "QPERC_12": -1, "QINTF_1": -1, "OFLOW_1": -1},
        }
    elif decisions["ARCH1"] == "tension1_1":
        upper = {
            "TENS_1": {"EFF_PPT": 1, "QSURF": -1, "EVAP_1": -1, "TENS2FREE_1": -1},
            "FREE_1": {"TENS2FREE_1": 1, "QPERC_12": -1, "QINTF_1": -1, "OFLOW_1": -1},
        }
    else:
        upper = {
            "WATR_1": {"EFF_PPT": 1, "QSURF": -1, "EVAP_1": -1, "QPERC_12": -1, "QINTF_1": -1, "OFLOW_1": -1},
        }

    if decisions["ARCH2"] == "tens2pll_2":
        lower = {
            "TENS_2": {"QPERC_12*(1-PERCFRAC)": 1, "EVAP_2": -1, "TENS2FREE_2": -1},
            "FREE_2A": {"QPERC_12*PERCFRAC/2": 1, "TENS2FREE_2/2": 1, "QBASE_2A": -1, "OFLOW_2A": -1},
            "FREE_2B": {"QPERC_12*PERCFRAC/2": 1, "TENS2FREE_2/2": 1, "QBASE_2B": -1, "OFLOW_2B": -1},
        }
    else:
        lower = {"WATR_2": {"QPERC_12": 1, "EVAP_2": -1, "QBASE_2": -1, "OFLOW_2": -1}}
    return {"upper": upper, "lower": lower}


@dataclass(frozen=True)
class StructureSpec:
    model_id: int
    decisions: dict[str, str]
    decision_codes: dict[str, int]
    state_names: tuple[str, ...]
    state_mask: dict[str, bool]
    parameter_names: tuple[str, ...]
    parameter_mask: dict[str, bool]
    flux_names: tuple[str, ...]
    topology: dict[str, Any]
    solver: dict[str, Any]

    @property
    def decision_vector(self) -> tuple[str, ...]:
        return tuple(self.decisions[name] for name in DECISION_ORDER)

    @property
    def decision_code_vector(self) -> tuple[int, ...]:
        return tuple(self.decision_codes[name] for name in DECISION_ORDER)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "decisions": dict(self.decisions),
            "decision_codes": dict(self.decision_codes),
            "decision_vector": list(self.decision_vector),
            "decision_code_vector": list(self.decision_code_vector),
            "state_names": list(self.state_names),
            "state_mask": dict(self.state_mask),
            "parameter_names": list(self.parameter_names),
            "parameter_mask": dict(self.parameter_mask),
            "flux_names": list(self.flux_names),
            "topology": self.topology,
            "solver": self.solver,
        }


def _from_row(row: Mapping[str, str]) -> StructureSpec:
    decisions = {name: row[name] for name in DECISION_ORDER}
    reason = _legal_reason(decisions)
    if reason:
        raise ValueError(f"illegal paper structure {row['ID']}: {reason}")
    state_names = _ARCH1_STATES[decisions["ARCH1"]] + _ARCH2_STATES[decisions["ARCH2"]]
    parameter_names = _parameter_names(decisions)
    return StructureSpec(
        model_id=int(row["ID"]),
        decisions=decisions,
        decision_codes={name: DECISION_CODES[value] for name, value in decisions.items()},
        state_names=state_names,
        state_mask={name: name in state_names for name in STATE_NAMES},
        parameter_names=parameter_names,
        parameter_mask={name: name in parameter_names for name in PARAMETER_NAMES},
        flux_names=tuple(dict.fromkeys(_ARCH1_FLUXES[decisions["ARCH1"]] + _ARCH2_FLUXES[decisions["ARCH2"]])),
        topology=_topology(decisions),
        solver=dict(SOLVER_CONFIG),
    )


_ROWS = _load_rows()
_SPECS = {spec.model_id: spec for spec in (_from_row(row) for row in _ROWS)}


def enumerate_structures() -> tuple[StructureSpec, ...]:
    """Return the paper's 78 structures in source-file order."""
    return tuple(_SPECS[int(row["ID"])] for row in _ROWS)


def get_structure(model_id: int | str) -> StructureSpec:
    """Resolve one paper model ID, preserving the paper's original ID."""
    try:
        return _SPECS[int(model_id)]
    except (KeyError, ValueError) as exc:
        raise KeyError(f"unknown FUSE-78 model ID: {model_id}") from exc


def default_parameters() -> dict[str, float]:
    """Return all extracted defaults, including inactive union coordinates."""
    return {name: float(meta["default"]) for name, meta in PARAMETERS.items()}


def parameter_metadata(name: str) -> Mapping[str, Any]:
    return PARAMETERS[name]


def validate_catalog() -> dict[str, Any]:
    """Validate count, legality, uniqueness, masks, and reverse mapping."""
    specs = enumerate_structures()
    ids = [spec.model_id for spec in specs]
    vectors = [spec.decision_vector for spec in specs]
    if len(specs) != 78 or len(set(ids)) != 78 or len(set(vectors)) != 78:
        raise AssertionError("paper catalogue is not 78 unique decision vectors")
    for spec in specs:
        if any(spec.decisions[key] != value for key, value in FIXED_PAPER_DECISIONS.items()):
            raise AssertionError(f"fixed decision drift in model {spec.model_id}")
        if len(spec.state_names) != sum(spec.state_mask.values()):
            raise AssertionError(f"state mask mismatch in model {spec.model_id}")
        if len(spec.parameter_names) != sum(spec.parameter_mask.values()):
            raise AssertionError(f"parameter mask mismatch in model {spec.model_id}")
        if set(spec.state_names) - set(STATE_NAMES):
            raise AssertionError(f"unknown state in model {spec.model_id}")
        if set(spec.parameter_names) - set(PARAMETER_NAMES):
            raise AssertionError(f"unknown parameter in model {spec.model_id}")
        if _legal_reason(spec.decisions):
            raise AssertionError(f"illegal model {spec.model_id}")

    theoretical = {
        (arch1, arch2, qsurf, qperc)
        for arch1 in _ARCH1_STATES
        for arch2 in _ARCH2_STATES if arch2 != "topmdexp_2"
        for qsurf in ("arno_x_vic", "prms_varnt", "tmdl_param")
        for qperc in ("perc_f2sat", "perc_w2sat", "perc_lower")
    }
    valid = {(s.decisions["ARCH1"], s.decisions["ARCH2"], s.decisions["QSURF"], s.decisions["QPERC"]) for s in specs}
    excluded = theoretical - valid
    if len(theoretical) != 108 or len(valid) != 78 or len(excluded) != 30:
        raise AssertionError("theoretical/valid structure counts drifted")
    return {
        "n_structures": len(specs),
        "n_unique_ids": len(set(ids)),
        "n_unique_decision_vectors": len(set(vectors)),
        "n_theoretical_open_combinations": len(theoretical),
        "n_excluded_combinations": len(excluded),
        "excluded_reasons": {
            "tension2_1+tens2pll_2": 9,
            "non-onestate_1+perc_w2sat": 24,
            "overlap": 3,
        },
        "state_union_size": len(STATE_NAMES),
        "parameter_union_size": len(PARAMETER_NAMES),
        "reverse_mapping_ok": all(get_structure(s.model_id).decision_vector == s.decision_vector for s in specs),
    }
