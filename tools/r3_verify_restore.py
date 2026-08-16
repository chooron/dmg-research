#!/usr/bin/env python3
"""Phase 2 verification of restored R3 data (truth + 6 misspec dPL runs).

Checks:
- truth npz shapes/keys; manifest fields; q_star finiteness; theta_star bounds
- truth vs dPL basin-ID set equality
- each dPL dir: COMPLETE marker, config protocol/target, 532-row summary,
  param shapes, seed; recompute val_kge median/mean
- write recovery manifest JSON
Exit 0 if all pass, 1 otherwise.
"""
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

RESULTS = Path("/home/jingxin/code/dmg-research/project/hydrodiag/results")
TRUTH = RESULTS / "r3_synthetic_truth_v1"
DPL_DIRS = {
    "base_42": RESULTS / "r3_misspec_dpl_xaj_seed_42",
    "base_123": RESULTS / "r3_misspec_dpl_xaj_seed_123",
    "base_2026": RESULTS / "r3_misspec_dpl_xaj_seed_2026",
    "tgd2_42": RESULTS / "r3_misspec_dpl_xaj_tgd2_seed_42",
    "tgd2_123": RESULTS / "r3_misspec_dpl_xaj_tgd2_seed_123",
    "tgd2_2026": RESULTS / "r3_misspec_dpl_xaj_tgd2_seed_2026",
}
HIST = {
    "base_42": 0.9081, "base_123": 0.9092, "base_2026": 0.9092,
    "tgd2_42": 0.9443, "tgd2_123": 0.9443, "tgd2_2026": 0.9443,
}

errors = []
report = {"results_root": str(RESULTS), "checks": {}}


def check(name, ok, detail=""):
    report["checks"][name] = {"pass": bool(ok), "detail": detail}
    if not ok:
        errors.append(name)


# ---------- truth ----------
z = np.load(TRUTH / "theta_star.npz")
q = np.load(TRUTH / "q_star.npz")
x = np.load(TRUTH / "x_star.npz")
sn = np.load(TRUTH / "snow_star.npz")
fs = np.load(TRUTH / "final_states.npz")
manifest = json.loads((TRUTH / "manifest.json").read_text())
gman = json.loads((TRUTH / "gstar_manifest.json").read_text())
gdiag = json.loads((TRUTH / "gstar_diagnostics.json").read_text())

check("truth_theta_shape", z["parameters"].shape == (531, 17))
check("truth_theta_names", list(z["parameter_names"]) ==
      ["cn_ctg", "cn_kf", "xaj_k", "xaj_b", "xaj_im", "xaj_um", "xaj_lm",
       "xaj_dm", "xaj_c", "xaj_sm", "xaj_ex", "xaj_ki", "xaj_kg", "xaj_ci",
       "xaj_cg", "xaj_a", "xaj_theta"])
check("truth_theta_basin_ids", len(z["basin_ids"]) == 531)
check("truth_q_shape", q["target_mm_day"].shape == (531, 12418))
check("truth_q_finite_nonneg",
      bool(np.isfinite(q["target_mm_day"]).all()) and bool((q["target_mm_day"] >= 0).all()))
check("truth_x_states", set(x.keys()) == {"wu", "wl", "wd", "s", "fr", "qi", "qg", "basin_ids"})
for k in ["wu", "wl", "wd", "s", "fr", "qi", "qg"]:
    check(f"truth_x_{k}_shape", x[k].shape == (531, 12418))
check("truth_snow_vars", set(sn.keys()) == {"G", "eTG", "sca", "rain", "melt", "effective_precip", "basin_ids"})
check("truth_final_states", fs["final_states"].shape == (531, 9) and fs["prod_final_states"].shape == (531, 9))
check("truth_roundtrip_zero", manifest["roundtrip"]["q_max_abs_diff_overall"] == 0.0)
check("truth_protocol", manifest["protocol"] == "r3_synthetic_truth_v1")
check("gstar_k15", gman["g_star"]["k"] == 15)
check("gstar_alpha", abs(gman["g_star"]["ridge_alpha"] - 316.22776601683796) < 1e-9)
check("gstar_frac_snow_idx", gman["inputs"]["frac_snow_index"] == 3)
check("gstar_n_basins", gman["inputs"]["n_basins"] == 531)
check("gdiag_cv_r2", abs(gdiag["attribute_mapping"]["cv_r2_total"] - 0.9391314435276207) < 1e-9)
check("gdiag_theta_eq_g", gdiag.get("theta_star_equals_g_star") is True)

truth_basin_set = set(str(b) for b in z["basin_ids"])
q_basin_set = set(str(b) for b in q["basin_ids"])
check("truth_theta_q_basin_ids_match", truth_basin_set == q_basin_set)

# ---------- dPL dirs ----------
dpl_report = {}
for key, d in DPL_DIRS.items():
    entry = {"path": str(d)}
    complete = (d / "COMPLETE").exists()
    entry["complete_marker"] = bool(complete)
    cfg_ok = (d / "config.json").exists()
    entry["config_exists"] = cfg_ok
    if cfg_ok:
        cfg = json.loads((d / "config.json").read_text())
        entry["protocol"] = cfg.get("_protocol")
        entry["model_name"] = cfg.get("model_name")
        entry["seed"] = cfg.get("training", {}).get("seed")
        entry["epochs"] = cfg.get("training", {}).get("epochs")
        entry["target_override"] = cfg.get("target_override_npz")
        entry["param_names"] = cfg.get("parameter_names")
        entry["n_basins_config"] = cfg.get("sampling_summary", {}).get("n_basins")
    # basin summary
    summary_ok = (d / "basin_final_summary.csv").exists()
    entry["summary_exists"] = summary_ok
    if summary_ok:
        with open(d / "basin_final_summary.csv") as f:
            rows = list(csv.DictReader(f))
        entry["summary_rows"] = len(rows)
        val = np.array([float(r["val_kge"]) for r in rows])
        entry["val_kge_median"] = float(np.median(val))
        entry["val_kge_mean"] = float(np.mean(val))
        entry["val_kge_min"] = float(np.min(val))
        basins = [r["basin_id"] for r in rows]
        entry["basin_id_set_matches_truth"] = set(basins) == truth_basin_set
    # param npz
    for npzname, expect_shape in [("best_parameters_physical.npz", (531, 15 if key.startswith("base") else 17)),
                                  ("best_parameters_normalized.npz", (531, 15 if key.startswith("base") else 17))]:
        p = d / npzname
        if p.exists():
            arr = np.load(p)["params"]
            entry[npzname.replace(".npz", "_shape")] = list(arr.shape)
    # historical comparison
    entry["hist_val_kge_median"] = HIST[key]
    entry["kge_match_hist"] = abs(entry.get("val_kge_median", -1) - HIST[key]) < 0.002
    check(f"dpl_{key}_complete", complete)
    check(f"dpl_{key}_protocol", entry.get("protocol") == "r3_misspec_dpl_synthetic_target_v1")
    check(f"dpl_{key}_seed", entry.get("seed") in (42, 123, 2026))
    check(f"dpl_{key}_epochs100", entry.get("epochs") == 100)
    check(f"dpl_{key}_rows531", entry.get("summary_rows") == 531)
    check(f"dpl_{key}_basins_match_truth", entry.get("basin_id_set_matches_truth") is True)
    check(f"dpl_{key}_target_qstar", str(entry.get("target_override")).endswith("q_star.npz"))
    check(f"dpl_{key}_kge_hist", entry.get("kge_match_hist") is True)
    dpl_report[key] = entry

report["dpl"] = dpl_report
report["truth"] = {
    "theta_shape": list(z["parameters"].shape),
    "q_shape": list(q["target_mm_day"].shape),
    "roundtrip": manifest["roundtrip"],
    "gstar": {"k": gman["g_star"]["k"], "alpha": gman["g_star"]["ridge_alpha"],
              "cv_r2": gdiag["attribute_mapping"]["cv_r2_total"]},
}
report["all_pass"] = not errors
out = RESULTS / "r3_recovery_manifest.json"
out.write_text(json.dumps(report, indent=1, default=str) + "\n")
print(f"ALL_PASS={report['all_pass']}  manifest -> {out}")
for name in errors:
    print("FAIL:", name)
# headline table
print("\n{:<10} {:>10} {:>10} {:>10} {}".format("run", "median", "mean", "hist", "match"))
for key, e in dpl_report.items():
    print("{:<10} {:>10.4f} {:>10.4f} {:>10.4f} {}".format(
        key, e.get("val_kge_median", float('nan')), e.get("val_kge_mean", float('nan')),
        e.get("hist_val_kge_median", float('nan')), "OK" if e.get("kge_match_hist") else "MISMATCH"))
sys.exit(0 if report["all_pass"] else 1)
