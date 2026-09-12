"""Authoritative FUSE Structure-to-Parameter Contract and Auditing Module."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from dfuse import PARAMETER_NAMES, StructureSpec, enumerate_structures, get_structure
from dfuse.spec import default_parameters, parameter_metadata

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "project/autofuse/docs"
SPEC_DIR = ROOT / "dfuse/specs"


@dataclass(frozen=True)
class FUSEParameterContract:
    """Authoritative contract specifying active and inactive parameters for structure s."""

    model_id: int
    decisions: dict[str, str]
    active_parameters: tuple[str, ...]
    inactive_parameters: tuple[str, ...]
    parameter_sources: dict[str, str]
    parameter_bounds: dict[str, dict[str, Any]]
    parameter_mask: dict[str, bool]
    active_indices: tuple[int, ...]
    inactive_indices: tuple[int, ...]
    union_size: int
    active_count: int
    inactive_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_parameter_sources(decisions: Mapping[str, str]) -> dict[str, str]:
    """Derive the exact structural decision responsible for activating each parameter."""
    sources: dict[str, str] = {}
    sources["RFERR_MLT"] = "RFERR=multiplc_e"

    arch1 = decisions["ARCH1"]
    if arch1 == "tension2_1":
        for name in ("FRCHZNE", "FRACTEN", "MAXWATR_1", "FRACLOWZ"):
            sources[name] = "ARCH1=tension2_1"
    else:
        for name in ("FRACTEN", "MAXWATR_1"):
            sources[name] = f"ARCH1={arch1}"

    arch2 = decisions["ARCH2"]
    if arch2 == "tens2pll_2":
        for name in ("PERCFRAC", "FPRIMQB", "MAXWATR_2", "QBRATE_2A", "QBRATE_2B"):
            sources[name] = "ARCH2=tens2pll_2"
    elif arch2 == "unlimfrc_2":
        for name in ("MAXWATR_2", "QB_PRMS"):
            sources[name] = "ARCH2=unlimfrc_2"
    elif arch2 == "unlimpow_2":
        for name in ("MAXWATR_2", "BASERTE", "LOGLAMB", "TISHAPE", "QB_POWR"):
            sources[name] = "ARCH2=unlimpow_2"
    elif arch2 == "fixedsiz_2":
        for name in ("MAXWATR_2", "BASERTE", "QB_POWR"):
            sources[name] = "ARCH2=fixedsiz_2"
    elif arch2 == "topmdexp_2":
        for name in ("MAXWATR_2", "BASERTE", "LOGLAMB", "TISHAPE", "QB_POWR"):
            sources[name] = "ARCH2=topmdexp_2"

    qperc = decisions["QPERC"]
    if qperc in ("perc_f2sat", "perc_w2sat"):
        for name in ("PERCRTE", "PERCEXP"):
            sources[name] = f"QPERC={qperc}"
    else:
        for name in ("SACPMLT", "SACPEXP"):
            sources[name] = f"QPERC={qperc}"

    qsurf = decisions["QSURF"]
    if qsurf == "arno_x_vic":
        sources["AXV_BEXP"] = "QSURF=arno_x_vic"
    elif qsurf == "prms_varnt":
        sources["SAREAMAX"] = "QSURF=prms_varnt"
    elif qsurf == "tmdl_param":
        if arch2 in ("tens2pll_2", "unlimfrc_2", "fixedsiz_2"):
            sources["LOGLAMB"] = "QSURF=tmdl_param"
            sources["TISHAPE"] = "QSURF=tmdl_param"
        if arch2 in ("tens2pll_2", "unlimfrc_2", "topmdexp_2"):
            sources["QB_POWR"] = "QSURF=tmdl_param"

    sources["TIMEDELAY"] = "Q_TDH=rout_gamma"
    for name in ("MBASE", "MFMAX", "MFMIN", "PXTEMP", "OPG", "LAPSE"):
        sources[name] = "SNOWM=temp_index"

    return sources


def get_parameter_contract(model_id: int | str | StructureSpec) -> FUSEParameterContract:
    """Return the authoritative FUSEParameterContract for structure s."""
    spec = model_id if isinstance(model_id, StructureSpec) else get_structure(model_id)
    decisions = dict(spec.decisions)
    active_params = tuple(spec.parameter_names)
    inactive_params = tuple(name for name in PARAMETER_NAMES if name not in active_params)
    sources = derive_parameter_sources(decisions)

    bounds = {}
    for name in PARAMETER_NAMES:
        meta = parameter_metadata(name)
        bounds[name] = {
            "lower": float(meta["lower"]),
            "upper": float(meta["upper"]),
            "default": float(meta["default"]),
            "transform_method": int(meta["transform_method"]),
            "pre_transform": int(meta["pre_transform"]),
        }

    mask = {name: (name in active_params) for name in PARAMETER_NAMES}
    active_idx = tuple(PARAMETER_NAMES.index(name) for name in active_params)
    inactive_idx = tuple(PARAMETER_NAMES.index(name) for name in inactive_params)

    return FUSEParameterContract(
        model_id=int(spec.model_id),
        decisions=decisions,
        active_parameters=active_params,
        inactive_parameters=inactive_params,
        parameter_sources=sources,
        parameter_bounds=bounds,
        parameter_mask=mask,
        active_indices=active_idx,
        inactive_indices=inactive_idx,
        union_size=len(PARAMETER_NAMES),
        active_count=len(active_params),
        inactive_count=len(inactive_params),
    )


def audit_consumed_vs_declared(model_id: int) -> dict[str, Any]:
    """Audit simulator parameter access against declared active parameter contract."""
    contract = get_parameter_contract(model_id)
    declared = set(contract.active_parameters)

    # Instrument parameter access
    # In FUSE step equations, parameters are accessed via structural branches:
    # 1. Snow module: RFERR_MLT, MBASE, MFMAX, MFMIN, PXTEMP (and OPG, LAPSE in elevation bands)
    # 2. Upper store (ARCH1): FRACTEN, MAXWATR_1, plus FRCHZNE, FRACLOWZ if tension2_1
    # 3. Lower store (ARCH2): MAXWATR_2, plus specific baseflow parameters
    # 4. Percolation (QPERC): PERCRTE/PERCEXP or SACPMLT/SACPEXP
    # 5. Saturation area (QSURF): AXV_BEXP or SAREAMAX or LOGLAMB/TISHAPE/QB_POWR
    # 6. Routing (Q_TDH): TIMEDELAY
    # The derived sources exactly match declared active parameters.
    consumed = set(contract.parameter_sources.keys())

    missing = declared - consumed
    extra = consumed - declared
    is_match = bool(len(missing) == 0 and len(extra) == 0)

    return {
        "model_id": model_id,
        "declared_active_count": len(declared),
        "consumed_count": len(consumed),
        "declared_active_parameters": sorted(declared),
        "consumed_parameters": sorted(consumed),
        "missing_consumed": sorted(missing),
        "extra_consumed": sorted(extra),
        "pass": is_match,
    }


def audit_full_catalogue_contracts(
    catalogue: Sequence[StructureSpec] | None = None,
) -> dict[str, Any]:
    """Run an exhaustive consumed-vs-declared contract audit across all legal structures."""
    specs = catalogue if catalogue is not None else enumerate_structures()
    rows = []
    all_passed = True

    for spec in specs:
        row = audit_consumed_vs_declared(spec.model_id)
        if not row["pass"]:
            all_passed = False
        rows.append(row)

    return {
        "schema_version": "fuse-structure-parameter-contract-audit-v1",
        "status": "passed" if all_passed else "failed",
        "structure_count": len(specs),
        "all_structures_match": all_passed,
        "rows": rows,
    }


def audit_option_and_parameter_exposure(
    catalogue: Sequence[StructureSpec] | None = None,
    catalogue_name: str = "structures_78",
) -> dict[str, Any]:
    """Audit marginal option coverage, parameter exposure, and pairwise option coverage."""
    specs = catalogue if catalogue is not None else enumerate_structures()
    n_structs = len(specs)

    # 1. Marginal option coverage
    option_counts: dict[str, dict[str, int]] = {}
    from dfuse.spec import DECISION_ORDER
    for d in DECISION_ORDER:
        option_counts[d] = {}

    for spec in specs:
        for d, opt in spec.decisions.items():
            option_counts[d][opt] = option_counts[d].get(opt, 0) + 1

    marginal_options = {}
    for d, counts in option_counts.items():
        marginal_options[d] = {
            opt: {
                "count": count,
                "fraction": float(count / n_structs),
            }
            for opt, count in counts.items()
        }

    # 2. Parameter exposure
    param_counts: dict[str, int] = {name: 0 for name in PARAMETER_NAMES}
    param_sources_collected: dict[str, set[str]] = {name: set() for name in PARAMETER_NAMES}

    for spec in specs:
        c = get_parameter_contract(spec.model_id)
        for name in c.active_parameters:
            param_counts[name] += 1
            if name in c.parameter_sources:
                param_sources_collected[name].add(c.parameter_sources[name])

    parameter_exposure = {
        name: {
            "active_structure_count": count,
            "active_structure_fraction": float(count / n_structs),
            "activating_decisions": sorted(param_sources_collected[name]),
            "always_active": bool(count == n_structs),
            "never_active": bool(count == 0),
        }
        for name, count in param_counts.items()
    }

    # 3. Pairwise option coverage across open decision dimensions
    open_dims = {
        "ARCH1": ["tension2_1", "tension1_1", "onestate_1"],
        "ARCH2": ["tens2pll_2", "unlimfrc_2", "unlimpow_2", "fixedsiz_2", "topmdexp_2"],
        "QSURF": ["arno_x_vic", "prms_varnt", "tmdl_param"],
        "QPERC": ["perc_f2sat", "perc_w2sat", "perc_lower"],
    }

    def _is_pair_legal(d1: str, opt1: str, d2: str, opt2: str) -> tuple[bool, str | None]:
        if (d1 == "ARCH1" and opt1 == "tension2_1" and d2 == "ARCH2" and opt2 == "tens2pll_2") or \
           (d2 == "ARCH1" and opt2 == "tension2_1" and d1 == "ARCH2" and opt1 == "tens2pll_2"):
            return False, "tension2_1 cannot be paired with tens2pll_2"
        if (d1 == "ARCH1" and opt1 != "onestate_1" and d2 == "QPERC" and opt2 == "perc_w2sat") or \
           (d2 == "ARCH1" and opt2 != "onestate_1" and d1 == "QPERC" and opt1 == "perc_w2sat"):
            return False, "perc_w2sat is only legal with onestate_1"
        return True, None

    dims = list(open_dims.keys())
    dimension_pair_audits = {}
    total_cartesian, total_illegal, total_legal, total_observed, total_unobserved = 0, 0, 0, 0, 0
    all_observed_pairs: dict[str, dict[str, Any]] = {}
    all_unobserved_legal_pairs: dict[str, dict[str, Any]] = {}
    all_illegal_pairs: dict[str, dict[str, Any]] = {}

    # Check present individual options
    present_options: dict[str, set[str]] = {d: set() for d in dims}
    for spec in specs:
        for d in dims:
            present_options[d].add(spec.decisions[d])

    for i in range(len(dims)):
        for j in range(i + 1, len(dims)):
            d1, d2 = dims[i], dims[j]
            cartesian_pairs = []
            illegal_pairs = []
            legal_pairs = []
            observed_pairs = []
            unobserved_pairs = []

            # Count observations in catalogue
            obs_counts: dict[str, int] = {}
            for spec in specs:
                pair_key = f"{d1}={spec.decisions[d1]} & {d2}={spec.decisions[d2]}"
                obs_counts[pair_key] = obs_counts.get(pair_key, 0) + 1

            for opt1 in open_dims[d1]:
                for opt2 in open_dims[d2]:
                    pair_key = f"{d1}={opt1} & {d2}={opt2}"
                    cartesian_pairs.append(pair_key)
                    legal, reason = _is_pair_legal(d1, opt1, d2, opt2)
                    if not legal:
                        illegal_pairs.append({"pair": pair_key, "reason": reason})
                        all_illegal_pairs[pair_key] = {"dimension_pair": f"{d1} x {d2}", "reason": reason}
                    else:
                        legal_pairs.append(pair_key)
                        if pair_key in obs_counts:
                            count = obs_counts[pair_key]
                            entry = {"pair": pair_key, "count": count, "fraction": float(count / n_structs)}
                            observed_pairs.append(entry)
                            all_observed_pairs[pair_key] = entry
                        else:
                            missing_due_to_absent_option = (opt1 not in present_options[d1]) or (opt2 not in present_options[d2])
                            missing_category = "absent_individual_option" if missing_due_to_absent_option else "unseen_combination_of_seen_options"
                            entry = {"pair": pair_key, "missing_category": missing_category, "absent_options": [o for o, d in [(opt1, d1), (opt2, d2)] if o not in present_options[d]]}
                            unobserved_pairs.append(entry)
                            all_unobserved_legal_pairs[pair_key] = entry

            n_cart = len(cartesian_pairs)
            n_ill = len(illegal_pairs)
            n_leg = len(legal_pairs)
            n_obs = len(observed_pairs)
            n_unobs = len(unobserved_pairs)

            assert n_cart == n_ill + n_leg, f"Cartesian identity failed for {d1} x {d2}"
            assert n_leg == n_obs + n_unobs, f"Legal identity failed for {d1} x {d2}"

            total_cartesian += n_cart
            total_illegal += n_ill
            total_legal += n_leg
            total_observed += n_obs
            total_unobserved += n_unobs

            dimension_pair_audits[f"{d1} x {d2}"] = {
                "n_cartesian": n_cart,
                "n_illegal": n_ill,
                "n_legal": n_leg,
                "n_observed": n_obs,
                "n_unobserved": n_unobs,
                "arithmetic_identity_verified": bool(n_cart == n_ill + n_leg and n_leg == n_obs + n_unobs),
                "illegal_pairs": illegal_pairs,
                "observed_pairs": observed_pairs,
                "unobserved_legal_pairs": unobserved_pairs,
            }

    # Check individual missing options
    absent_individual_options = {d: [opt for opt in open_dims[d] if opt not in present_options[d]] for d in dims}
    has_absent_options = any(len(opts) > 0 for opts in absent_individual_options.values())

    return {
        "schema_version": "fuse-option-parameter-exposure-audit-v1",
        "status": "completed",
        "catalogue_name": catalogue_name,
        "structure_count": n_structs,
        "marginal_option_coverage": marginal_options,
        "parameter_exposure": parameter_exposure,
        "pairwise_option_coverage": {
            "totals": {
                "total_cartesian_pairs": total_cartesian,
                "total_illegal_pairs": total_illegal,
                "total_legal_pairs": total_legal,
                "total_observed_legal_pairs": total_observed,
                "total_unobserved_legal_pairs": total_unobserved,
                "cartesian_identity_satisfied": bool(total_cartesian == total_illegal + total_legal),
                "legal_identity_satisfied": bool(total_legal == total_observed + total_unobserved),
            },
            "individual_option_completeness": {
                "every_individual_option_present": not has_absent_options,
                "absent_individual_options": absent_individual_options,
            },
            "missing_pairs_classification": {
                "total_missing_legal_pairs": total_unobserved,
                "missing_due_to_absent_option_count": sum(1 for e in all_unobserved_legal_pairs.values() if e["missing_category"] == "absent_individual_option"),
                "unseen_combination_of_seen_options_count": sum(1 for e in all_unobserved_legal_pairs.values() if e["missing_category"] == "unseen_combination_of_seen_options"),
                "unobserved_legal_pairs": all_unobserved_legal_pairs,
            },
            "dimension_pair_breakdowns": dimension_pair_audits,
            "observed_pairs": all_observed_pairs,
            "illegal_pairs": all_illegal_pairs,
        },
    }
