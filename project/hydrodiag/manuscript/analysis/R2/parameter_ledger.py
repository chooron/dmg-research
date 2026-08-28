"""Build and validate the raw long-form parameter ledger from lowest-level artifacts.

Ledger schema:
  basin_id x paradigm x structure x member_id x parameter
  Total expected rows:
    - IC:  531 basins x 3 structures x 10 starts x 15 parameters = 238,950 rows
    - dPL: 531 basins x 3 structures x 3 seeds   x 15 parameters =  71,685 rows
    - Total: 310,635 rows
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from r2_config import (
    CANONICAL_R1_BASIN_TABLE,
    DPL_SEED_DIRS,
    DPL_SEEDS,
    IC_RAW_DIRS,
    IC_STARTS,
    PARADIGMS,
    RESULTS_DIR,
    SNOW_FILE,
    STRUCTURES,
    TOTAL_BASINS,
)
from shared_parameter_specs import (
    PARAMETER_METADATA,
    SHARED_15_PARAMETERS,
    STRUCTURE_PARAM_LAYOUTS,
)


def load_canonical_snow_metadata() -> Dict[str, Tuple[float, str]]:
    """Load verified basin_id -> (frac_snow, snow_stratum) from canonical R1 outputs."""
    if CANONICAL_R1_BASIN_TABLE.exists():
        df = pd.read_csv(CANONICAL_R1_BASIN_TABLE, usecols=["basin_id", "frac_snow", "snow_stratum"])
        df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
        df = df.drop_duplicates("basin_id")
        return {r["basin_id"]: (float(r["frac_snow"]), str(r["snow_stratum"])) for _, r in df.iterrows()}

    df = pd.read_csv(SNOW_FILE)
    df["basin_id"] = df["basin_id"].astype(str).str.zfill(8)
    return {r["basin_id"]: (float(r["frac_snow"]), str(r["snow_stratum"])) for _, r in df.iterrows()}


def build_raw_parameter_ledger(
    output_dir: Path | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build long-form parameter ledger from lowest-level IC raw JSONs and dPL physical parameter NPZs.

    Returns:
        (ledger_rows, audit_summary)
    """
    out_dir = output_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    snow_meta = load_canonical_snow_metadata()
    basins = sorted(snow_meta.keys())
    if len(basins) != TOTAL_BASINS:
        raise RuntimeError(f"Expected {TOTAL_BASINS} unique basins, got {len(basins)}")

    ledger_rows: List[Dict[str, Any]] = []
    seen_keys = set()

    # -------------------------------------------------------------
    # 1. IC: Read 10 raw JSON restart files per basin per structure
    # -------------------------------------------------------------
    ic_audit = {}
    for struct in STRUCTURES:
        raw_dir = IC_RAW_DIRS[struct]
        if not raw_dir.exists():
            raise FileNotFoundError(f"Missing IC raw directory for {struct}: {raw_dir}")

        json_files = sorted(raw_dir.glob("*.json"))
        records_by_basin = defaultdict(dict)

        for p in json_files:
            data = json.loads(p.read_text(encoding="utf-8"))
            b_id = str(data.get("basin_id", "")).zfill(8)
            if b_id not in snow_meta:
                continue
            start_idx = int(data.get("start", -1))
            status = data.get("status", "")
            train_kge = float(data.get("train_metrics", {}).get("kge", np.nan))
            test_kge = float(data.get("test_metrics", {}).get("kge", np.nan))
            p_names = data.get("parameter_names", [])
            p_values = data.get("parameters", [])

            p_map = dict(zip(p_names, p_values))
            records_by_basin[b_id][start_idx] = {
                "params": p_map,
                "train_kge": train_kge,
                "test_kge": test_kge,
                "train_objective": float(data.get("train_objective", train_kge)),
                "status": status,
                "path": str(p),
                "model": data.get("model", f"XAJ_{struct}"),
            }

        # Check completeness
        missing_basins = [b for b in basins if len(records_by_basin[b]) != len(IC_STARTS)]
        if missing_basins:
            raise RuntimeError(f"IC {struct} has incomplete starts for basins: {missing_basins[:5]}")

        ic_audit[struct] = {
            "total_files": len(json_files),
            "complete_basins": len(basins),
            "starts_per_basin": len(IC_STARTS),
        }

        # Emit rows for shared 15 parameters
        for b_id in basins:
            frac_snow, stratum = snow_meta[b_id]
            for start_idx in IC_STARTS:
                rec = records_by_basin[b_id][start_idx]
                p_map = rec["params"]
                member_id = f"start_{start_idx:02d}"

                for p_name in SHARED_15_PARAMETERS:
                    if p_name not in p_map:
                        raise KeyError(f"Missing parameter {p_name} in IC {struct} {b_id} {member_id}")

                    phys_val = float(p_map[p_name])
                    lo = PARAMETER_METADATA[p_name]["lower"]
                    hi = PARAMETER_METADATA[p_name]["upper"]
                    norm_val = (phys_val - lo) / (hi - lo)

                    key = (b_id, "IC", struct, member_id, p_name)
                    if key in seen_keys:
                        raise RuntimeError(f"Duplicate ledger key: {key}")
                    seen_keys.add(key)

                    ledger_rows.append({
                        "basin_id": b_id,
                        "paradigm": "IC",
                        "regime": "IC",
                        "structure": struct,
                        "member_id": member_id,
                        "start_or_seed": start_idx,
                        "parameter": p_name,
                        "symbol": PARAMETER_METADATA[p_name]["symbol"],
                        "physical_value": phys_val,
                        "lower_bound": lo,
                        "upper_bound": hi,
                        "normalized_value": norm_val,
                        "train_kge": rec["train_kge"],
                        "test_kge": rec["test_kge"],
                        "train_objective": rec["train_objective"],
                        "frac_snow": frac_snow,
                        "snow_stratum": stratum,
                        "source_file": rec["path"],
                        "model_identity": rec["model"],
                    })

    # -------------------------------------------------------------
    # 2. dPL: Read 3 physical parameter NPZ files per structure
    # -------------------------------------------------------------
    dpl_audit = {}
    for struct in STRUCTURES:
        struct_dir = DPL_SEED_DIRS[struct]
        layout = STRUCTURE_PARAM_LAYOUTS[struct]
        dpl_audit[struct] = {"seeds": []}

        for seed in DPL_SEEDS:
            seed_dir = struct_dir / f"seed_{seed}" if struct != "TGD" else struct_dir / f"seed_{seed}"
            npz_path = seed_dir / "best_parameters_physical.npz"
            if not npz_path.exists():
                raise FileNotFoundError(f"Missing dPL physical parameter file: {npz_path}")

            cfg_path = seed_dir / "config.json"
            model_ident = f"dPL_XAJ_{struct}"
            if cfg_path.exists():
                cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
                model_ident = cfg.get("model", {}).get("name", model_ident)

            data = np.load(npz_path)
            params_mat = data["params"]
            if np.isnan(params_mat).any():
                # Reconstruct clean 531 parameters from best_checkpoint.pt if the npz has evaluation-omitted NaNs
                import sys, torch
                from r2_config import PROJECT_ROOT, DATA_DIR, BASIN_FILE
                sys.path.insert(0, str(PROJECT_ROOT))
                from training.dpl.run_dpl_model import LITE_MODEL_REGISTRY, StaticParameterNet, physical_parameters, robust_normalize
                from ablation.ic_core.data_adapter import load_531_bundle

                bundle = load_531_bundle({
                    "project_root": str(PROJECT_ROOT),
                    "dataset_path": str(DATA_DIR / "camels_dataset"),
                    "gage_ids_path": str(DATA_DIR / "gage_id.npy"),
                    "dates_path": str(DATA_DIR / "camels_dates.npy"),
                    "basin_list_path": str(BASIN_FILE),
                    "periods": {
                        "warmup": {"start": "1980-10-01", "end": "1981-09-30"},
                        "train": {"start": "1981-10-01", "end": "1995-09-30"},
                        "test": {"start": "1995-10-01", "end": "2010-09-30"}
                    }
                })
                all_attrs, _ = robust_normalize(bundle.raw_attributes.astype(np.float32))
                model_key = "XAJ" if struct == "Base" else ("XAJ_CN" if struct == "CN" else "XAJ_TGD2")
                model_cls, specs = LITE_MODEL_REGISTRY[model_key]
                names = list(specs)
                hidden = [cfg.get("network", {}).get("hidden_size", 64)] * cfg.get("network", {}).get("depth", 2)
                net = StaticParameterNet(all_attrs.shape[1], specs, hidden, cfg.get("network", {}).get("dropout", 0.0), cfg.get("network", {}).get("output_epsilon", 1e-4)).eval()
                ckpt = torch.load(seed_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
                net.load_state_dict(ckpt["state_dict"])
                lower_t = torch.tensor([specs[n]["lower"] for n in names], dtype=torch.float32)
                ranges_t = torch.tensor([specs[n]["upper"] - specs[n]["lower"] for n in names], dtype=torch.float32)
                with torch.no_grad():
                    theta = net(torch.from_numpy(all_attrs))
                    phys_dict = physical_parameters(theta, names, lower_t, ranges_t)
                params_mat = np.stack([np.asarray(phys_dict[n]) for n in names], axis=1)

            if params_mat.shape != (TOTAL_BASINS, layout["total_params"]):
                raise RuntimeError(
                    f"dPL {struct} seed {seed} shape {params_mat.shape} != expected ({TOTAL_BASINS}, {layout['total_params']})"
                )

            dpl_audit[struct]["seeds"].append(seed)
            member_id = f"seed_{seed}"

            for b_idx, b_id in enumerate(basins):
                frac_snow, stratum = snow_meta[b_id]

                for p_name in SHARED_15_PARAMETERS:
                    col_idx = layout["shared_indices"][p_name]
                    phys_val = float(params_mat[b_idx, col_idx])
                    lo = PARAMETER_METADATA[p_name]["lower"]
                    hi = PARAMETER_METADATA[p_name]["upper"]
                    norm_val = (phys_val - lo) / (hi - lo)

                    key = (b_id, "dPL", struct, member_id, p_name)
                    if key in seen_keys:
                        raise RuntimeError(f"Duplicate ledger key: {key}")
                    seen_keys.add(key)

                    ledger_rows.append({
                        "basin_id": b_id,
                        "paradigm": "dPL",
                        "regime": "dPL",
                        "structure": struct,
                        "member_id": member_id,
                        "start_or_seed": seed,
                        "parameter": p_name,
                        "symbol": PARAMETER_METADATA[p_name]["symbol"],
                        "physical_value": phys_val,
                        "lower_bound": lo,
                        "upper_bound": hi,
                        "normalized_value": norm_val,
                        "train_kge": np.nan,
                        "test_kge": np.nan,
                        "train_objective": np.nan,
                        "frac_snow": frac_snow,
                        "snow_stratum": stratum,
                        "source_file": str(npz_path),
                        "model_identity": model_ident,
                    })

    # Verification of total row count
    expected_ic_rows = TOTAL_BASINS * len(STRUCTURES) * len(IC_STARTS) * len(SHARED_15_PARAMETERS)
    expected_dpl_rows = TOTAL_BASINS * len(STRUCTURES) * len(DPL_SEEDS) * len(SHARED_15_PARAMETERS)
    expected_total_rows = expected_ic_rows + expected_dpl_rows

    if len(ledger_rows) != expected_total_rows:
        raise RuntimeError(f"Ledger row count {len(ledger_rows)} != expected {expected_total_rows}")

    # Write to CSV
    fields = [
        "basin_id", "paradigm", "regime", "structure", "member_id", "start_or_seed",
        "parameter", "symbol", "physical_value", "lower_bound", "upper_bound",
        "normalized_value", "train_kge", "test_kge", "train_objective",
        "frac_snow", "snow_stratum", "source_file", "model_identity"
    ]

    out_file = out_dir / "raw_parameter_ledger.csv"
    with out_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in ledger_rows:
            writer.writerow(r)

    audit_summary = {
        "status": "PASS",
        "total_rows": len(ledger_rows),
        "expected_rows": expected_total_rows,
        "ic_audit": ic_audit,
        "dpl_audit": dpl_audit,
        "total_basins": TOTAL_BASINS,
        "structures": list(STRUCTURES),
        "paradigms": list(PARADIGMS),
        "parameters_count": len(SHARED_15_PARAMETERS),
        "output_file": str(out_file),
    }

    with (out_dir / "raw_parameter_ledger_audit.json").open("w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)

    return ledger_rows, audit_summary


if __name__ == "__main__":
    rows, audit = build_raw_parameter_ledger()
    print(f"Raw parameter ledger built successfully: {len(rows)} rows (Expected: {audit['expected_rows']}).")
