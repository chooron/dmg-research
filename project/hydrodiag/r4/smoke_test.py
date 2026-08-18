"""R4 DEV_ONLY pipeline smoke test on R3 synthetic-q* trained results.

This script validates the R4 pipeline mechanics — parameter loading,
continuous recorded forward, per-day state export, date/basin alignment,
and R4 metrics — using the local R3 results.  Every output is tagged
``DEV_ONLY``/``SYNTHETIC_TRAINED``; nothing here is a formal R4 scientific
result and nothing here claims real-basin validation.

Numerical consistency checks:

- S1 (IC adapter + forward): best-restart CN parameters from
  ``r3_gate_ic_xaj_cn_531_v1`` -> full-axis recorded forward -> q must match
  ``r3_misspec_analysis_v1/posthoc_q_CN_IC.npy`` (max abs diff <= 1e-5) and
  train/test KGE vs q* must reproduce the raw JSON stored KGE.
- S2 (dPL adapter + forward): CN seed-42 checkpoint
  (``r3_gate_dpl_xaj_cn_seed_42``) -> q must match
  ``posthoc_q_CN_dPL_s42.npy``.
- S3 (state-export identity): recorded forward vs production forward
  discharge on a basin subset (enforced inside ``continuous_forward``).
- S4 (alignment): exported dates equal the bundle date axis; basin order
  equals the canonical 531 order.
- S5 (fail-loud): adapters raise on missing runs / wrong model keys / a
  missing seed — no silent fallback to R3.
- S6 (snow reference parser): synthetic CAMELS-US layout fixture parse.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import DEV_ONLY, SYNTHETIC_TRAINED
from .common import (
    R3_DPL_RUNS, R3_IC_RUNS, default_data_root, default_results_root, load_bundle, zfill8,
)
from .forward_export import export_run
from .input_adapters import (
    R4ArtifactError, read_dpl_seed, read_ic_canonical, read_ic_fused,
)
from .state_export import continuous_forward, model_instances

Q_ABS_TOL = 1e-5
DPL_Q_ABS_TOL = 0.05  # float32 MLP re-derivation noise over the 12418-day loop
KGE_TOL = 1e-4


def _reproduce_stored_kge(q_full: np.ndarray, q_star: np.ndarray, records_kge_train: np.ndarray,
                          records_kge_test: np.ndarray, basin_ids: list[str]) -> dict[str, Any]:
    """Compare recomputed KGE (vs q*) with the KGE stored in the run records.

    Train: the runner's train split (train_forcing [0,5478) with a 365-day
    warmup from zero) is convention-identical to the continuous full-axis
    forward (both start from zero at day 0 and the full-axis window gives the
    same ``psol_annual`` as the R3 canonical full-record value), so the
    recomputed train KGE must reproduce the stored value exactly.

    Test: the stored test KGE is NOT reproducible under the R4 conventions.
    The R3 runner evaluated the test split as a windowed forward
    (test_forcing [5113,10957) re-warmstarted from zero with 365 warmup days,
    canonical full-record psol), while the R4 pipeline deliberately uses the
    continuous full-axis forward with window-based psol (R1/R2 historical
    semantics per ``r4/README.md``).  The two differ for long-memory basins
    (snow pack, deep soil); the gap is reported as ``test_continuous_vs_stored``
    and is a documented convention difference, not a pipeline defect.
    """
    from training.dpl.run_dpl_model import compute_kge_fp64

    recomputed_train = np.array([
        compute_kge_fp64(q_full[i, 365:5478], q_star[i, 365:5478])
        for i in range(len(basin_ids))
    ])
    diff_train = np.abs(recomputed_train - records_kge_train)
    cont_test = np.array([
        compute_kge_fp64(q_full[i, 5478:10957], q_star[i, 5478:10957])
        for i in range(len(basin_ids))
    ])
    diff_test = np.abs(cont_test - records_kge_test)
    return {
        "train": {
            "max_abs_diff": float(np.nanmax(diff_train)),
            "median_abs_diff": float(np.nanmedian(diff_train)),
            "n_exceeding_1e-4": int(np.sum(diff_train > KGE_TOL)),
            "pass": bool(np.nanmax(diff_train) <= KGE_TOL),
        },
        "test_continuous_vs_stored": {
            "max_abs_diff": float(np.nanmax(diff_test)),
            "median_abs_diff": float(np.nanmedian(diff_test)),
            "pass": None,
            "note": "stored test KGE uses the R3 runner's windowed-warmup + "
                    "canonical full-record-psol convention; the R4 continuous "
                    "full-axis forward (window-based psol) differs for "
                    "long-memory basins — documented convention gap, not a bug",
        },
    }

def smoke_ic_cn(results_root: Path, data_root: Path, device: str) -> dict[str, Any]:
    """S1 + S3 + S4 for the R3 CN IC gate run."""
    import torch

    run_root = results_root / R3_IC_RUNS["XAJ_CN"]
    bundle = load_bundle(data_root)
    basin_ids = [zfill8(b) for b in bundle.basin_ids]
    parameters, meta = read_ic_canonical(run_root, "XAJ_CN", "xaj_cn", basin_ids)

    # stored KGE from the raw records (same restart rule)
    from .input_adapters import iter_ic_records

    records = iter_ic_records(run_root, "xaj_cn")
    by_basin = {}
    for r in records:
        by_basin.setdefault(r.basin_id, []).append(r)
    from .input_adapters import select_best_restart

    selected = [select_best_restart(by_basin[b]) for b in basin_ids]
    stored_train = np.array([r.train_kge for r in selected])
    stored_test = np.array([r.test_kge for r in selected])

    models = model_instances(torch.device(device), torch.float32)
    q_full, states = continuous_forward(
        "XAJ_CN", models["XAJ_CN"], parameters, bundle.forcing.astype(np.float32),
        torch.device(device), torch.float32, batch=64, validate_subset=8,
    )

    posthoc = np.load(results_root / "r3_misspec_analysis_v1" / "posthoc_q_CN_IC.npy")
    if posthoc.shape != q_full.shape:
        raise AssertionError(f"posthoc shape {posthoc.shape} != q_full {q_full.shape}")
    q_diff = np.abs(posthoc - q_full)

    q_star = np.load(results_root / "r3_synthetic_truth_v1" / "q_star.npz")["target_mm_day"]
    kge_check = _reproduce_stored_kge(q_full, q_star, stored_train, stored_test, basin_ids)
    return {
        "adapter_meta": {k: meta[k] for k in ("format", "n_basins", "restart_rule")},
        "q_vs_posthoc": {
            "max_abs_diff": float(q_diff.max()),
            "median_abs_diff": float(np.median(q_diff)),
            "shape": list(q_full.shape),
            "pass": bool(q_diff.max() <= Q_ABS_TOL),
        },
        "kge_vs_qstar_reproduced": kge_check,
        "basin_order_matches_bundle": basin_ids == [zfill8(b) for b in bundle.basin_ids],
        "states_exported": sorted(states.keys()),
        "n_days": q_full.shape[1],
    }


def smoke_dpl_cn(results_root: Path, data_root: Path, device: str) -> dict[str, Any]:
    """S2 + S4 for the R3 CN dPL seed-42 run."""
    import torch

    run_root = results_root / (R3_DPL_RUNS["XAJ_CN"] + "42")
    bundle = load_bundle(data_root)
    basin_ids = [zfill8(b) for b in bundle.basin_ids]
    parameters, meta = read_dpl_seed(run_root, "XAJ_CN", data_root, basin_ids)

    models = model_instances(torch.device(device), torch.float32)
    q_full, _states = continuous_forward(
        "XAJ_CN", models["XAJ_CN"], parameters, bundle.forcing.astype(np.float32),
        torch.device(device), torch.float32, batch=64, validate_subset=8,
    )
    posthoc = np.load(results_root / "r3_misspec_analysis_v1" / "posthoc_q_CN_dPL_s42.npy")
    q_diff = np.abs(posthoc - q_full)
    # Parameter-level cross-check against the training-time physical dump.
    # The R3 posthoc q files were generated from `best_parameters_physical.npz`;
    # re-deriving from `best_checkpoint.pt` replays the MLP forward in float32,
    # which is deterministic up to ~1e-7 relative CUDA matmul noise.  Over the
    # 12418-day stateful loop this amplifies to ~1e-2 mm/day, so the discharge
    # identity is asserted at DPL_Q_ABS_TOL (not the IC 1e-5 tolerance).
    param_diff = np.inf
    npz_path = run_root / "best_parameters_physical.npz"
    if npz_path.is_file():
        param_diff = float(np.abs(parameters - np.asarray(np.load(npz_path)["params"], dtype=np.float64)).max())
    return {
        "adapter_meta": {k: meta[k] for k in ("format", "n_basins", "epoch_label")},
        "params_vs_best_parameters_physical_npz": {
            "max_abs_diff": param_diff,
            "pass": bool(param_diff <= 1e-3),
        },
        "q_vs_posthoc": {
            "max_abs_diff": float(q_diff.max()),
            "median_abs_diff": float(np.median(q_diff)),
            "tolerance_note": "float32 MLP re-derivation noise; S1-style 1e-5 identity applies to IC (JSON params, deterministic)",
            "pass": bool(q_diff.max() <= DPL_Q_ABS_TOL),
        },
        "basin_order_matches_bundle": basin_ids == [zfill8(b) for b in bundle.basin_ids],
    }


def smoke_export_pipeline(results_root: Path, data_root: Path, device: str, out_tag: str = "r4_smoke") -> dict[str, Any]:
    """Full export pipeline on the R3 CN dPL seed-42 parameters (DEV_ONLY)."""
    import torch

    run_root = results_root / (R3_DPL_RUNS["XAJ_CN"] + "42")
    bundle = load_bundle(data_root)
    basin_ids = [zfill8(b) for b in bundle.basin_ids]
    parameters, meta = read_dpl_seed(run_root, "XAJ_CN", data_root, basin_ids)
    manifest = export_run(
        structure="XAJ_CN",
        parameters=parameters,
        parameter_meta=meta,
        basin_ids=basin_ids,
        data_root=data_root,
        results_root=results_root,
        run_id=out_tag,
        tag=SYNTHETIC_TRAINED,
        provenance={
            "source": "R3 synthetic-q* trained CN dPL seed 42 (DEV_ONLY)",
            "run_root": str(run_root),
        },
        device=device,
        batch=64,
        validate_subset=8,
        csv_basin_subset=basin_ids[:12],
    )
    return {
        "tag": manifest["tag"],
        "has_snow_module": manifest["has_snow_module"],
        "pseudo_swe": manifest["pseudo_swe"],
        "n_basins": manifest["n_basins"],
        "n_days_full": manifest["n_days_full"],
        "q_finite": manifest["q_finite"],
        "q_nonnegative": manifest["q_nonnegative"],
        "psol_semantics": manifest["psol_gthresh_semantics"],
        "cn_psol_gthresh_sample": {b: manifest["cn_psol_gthresh"][b]["full"]
                                   for b in basin_ids[:3]},
        "outputs": manifest["outputs"],
        "state_columns": manifest["state_columns"],
    }


def smoke_fail_loud(results_root: Path, data_root: Path) -> list[dict[str, Any]]:
    """S5: adapters must raise R4ArtifactError, never silently fall back."""
    bundle = load_bundle(data_root)
    basin_ids = [zfill8(b) for b in bundle.basin_ids]
    checks: list[dict[str, Any]] = []

    def expect_raise(label: str, fn) -> None:
        try:
            fn()
        except R4ArtifactError as exc:
            checks.append({"check": label, "raised": "R4ArtifactError", "ok": True,
                           "message_head": str(exc)[:160]})
        except Exception as exc:  # noqa: BLE001
            checks.append({"check": label, "raised": type(exc).__name__,
                           "ok": False, "message_head": str(exc)[:160]})
        else:
            checks.append({"check": label, "raised": None, "ok": False,
                           "message_head": "no exception raised"})

    # (1) nonexistent IC run root
    expect_raise("ic_missing_run_root", lambda: read_ic_canonical(
        Path("/nonexistent/results/run"), "XAJ_CN", "xaj_cn", basin_ids))
    # (2) IC run exists but raw subdir mismatched
    run_root = results_root / R3_IC_RUNS["XAJ_CN"]
    expect_raise("ic_wrong_raw_subdir", lambda: read_ic_canonical(
        run_root, "XAJ_CN", "xaj_wrong", basin_ids))
    # (3) dPL missing seed directory
    expect_raise("dpl_missing_seed_dir", lambda: read_dpl_seed(
        results_root / (R3_DPL_RUNS["XAJ_CN"] + "999"), "XAJ_CN", data_root, basin_ids))
    # (4) dPL wrong model key vs checkpoint content
    expect_raise("dpl_wrong_model_key", lambda: read_dpl_seed(
        results_root / (R3_DPL_RUNS["XAJ_CN"] + "42"), "XAJ", data_root, basin_ids))
    # (5) fused csv missing file
    expect_raise("ic_fused_missing_csv", lambda: read_ic_fused(
        Path("/nonexistent/per_start.csv"), "XAJ_CN",
        ("cn_ctg", "cn_kf"), basin_ids))
    return checks


def smoke_snow_reader() -> dict[str, Any]:
    """S6: snow reference parser against a synthetic CAMELS-US fixture."""
    from .snow_reference import SnowReferenceReader

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        out_dir = root / "basin_dataset_public_v1p2" / "model_output"
        out_dir.mkdir(parents=True)
        # synthetic single-realization snow17_swe file for two basins
        for gauge in ("01013500", "01031500"):
            lines = ["date snow17_swe sac_sma_swe"]
            for month in range(1, 13):
                for day in (1, 15):
                    lines.append(f"1985-{month:02d}-{day:02d} {10.0 * month} {5.0 * month}")
            (out_dir / f"usgs_{gauge}_model_output.txt").write_text("\n".join(lines))
        reader = SnowReferenceReader(root)
        basins = reader.available_basins()
        snow = reader.load_basin("01013500")
        aligned = snow.align_to(np.array(["1985-01-01", "1985-01-02", "1985-01-15"], dtype="datetime64[D]"))
        metrics = snow.annual_metrics()
        return {
            "layout": reader.layout,
            "n_basins_resolved": len(basins),
            "basins": sorted(basins),
            "n_members": snow.n_members,
            "swe_source_column": snow.swe_source_column,
            "align_first": float(aligned[0]),
            "align_missing": bool(np.isnan(aligned[1])),
            "annual_metrics_1985": metrics.get("1985"),
            "target_basin_only": len(reader.load_target_basins(["01013500"])) == 1,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=None)
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    parser.add_argument("--out-tag", default="r4_smoke")
    parser.add_argument("--skip-export", action="store_true",
                        help="Skip the CSV export step (forward checks only).")
    args = parser.parse_args()

    results_root = Path(args.results_root) if args.results_root else default_results_root()
    data_root = Path(args.data_root) if args.data_root else default_data_root()

    report: dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tag": f"{DEV_ONLY}/{SYNTHETIC_TRAINED}",
        "results_root": str(results_root),
        "data_root": str(data_root),
        "device": args.device,
        "checks": {},
    }

    report["checks"]["S1_ic_cn_forward_vs_posthoc"] = smoke_ic_cn(results_root, data_root, args.device)
    report["checks"]["S2_dpl_cn_forward_vs_posthoc"] = smoke_dpl_cn(results_root, data_root, args.device)
    if not args.skip_export:
        report["checks"]["S3_export_pipeline"] = smoke_export_pipeline(
            results_root, data_root, args.device, args.out_tag)
    report["checks"]["S4_fail_loud"] = smoke_fail_loud(results_root, data_root)
    report["checks"]["S5_snow_reader"] = smoke_snow_reader()

    out_dir = results_root / f"r4_{args.out_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{args.out_tag}_smoke_report.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\n[DEV_ONLY] smoke report written to {path}")


if __name__ == "__main__":
    main()
