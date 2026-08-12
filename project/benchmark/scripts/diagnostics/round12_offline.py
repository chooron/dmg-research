"""Round-12 offline tables and K1 matrix from immutable experiment records."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/dpl_round12_20260805"
OUT.mkdir(parents=True, exist_ok=True)

AUTO = ROOT / "results/dpl_full_retrain_20260804/auto100"
EPOCHS = pd.read_csv(AUTO / "epochs.csv")
HEALTH = pd.read_csv(AUTO / "health.csv")
GRADS = pd.read_csv(AUTO / "parameter_gradients.csv")
CMA = pd.read_csv(ROOT / "results/full300_final_36models_evaluation/full300_kge_model_summary.csv")
MAPPING = pd.read_csv(ROOT / "results/dpl_training_pilot_20260801/i2_mapping/i2_model_summary.csv")

L1_MODELS = ["collie1", "gr4j", "hillslope", "penman", "tcm"]
EDGE_MODELS = [
    "australia", "collie2", "collie3", "flexb", "flexi", "flexis", "hbv96",
    "modhydrolog", "mopex1", "mopex4", "mopex5", "newzealand1", "newzealand2",
    "penman", "plateau", "tcm", "topmodel", "vic", "wetland",
]


def write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, float_format="%.8f")


def l1() -> pd.DataFrame:
    rows = []
    for model in L1_MODELS:
        x = EPOCHS[EPOCHS.model == model].sort_values("epoch")
        first = float(x.iloc[0].validation_median_kge)
        best = float(x.validation_median_kge.max())
        best_epoch = int(x.loc[x.validation_median_kge.idxmax(), "epoch"])
        c = CMA[CMA.model == model]
        cma = float(c.iloc[0].test_kge_median) if len(c) else None
        gap = best - cma if cma is not None else None
        if cma is None:
            verdict = "不可判定（无CMA归档）"
        elif gap <= 0.05:
            verdict = "无问题（起点已高/已接近CMA）"
        elif gap > 0.10 and best - first < 0.05:
            verdict = "确实学不动"
        else:
            verdict = "不属于两类明确异常"
        rows.append({"model": model, "epoch1_median_kge": first, "best_median_kge": best,
                     "best_epoch": best_epoch, "absolute_improvement": best-first,
                     "cma_reference": cma, "best_minus_cma": gap, "verdict": verdict})
    out = pd.DataFrame(rows)
    write(out, "l1_effective_learning.csv")
    return out


def mapping_table() -> pd.DataFrame:
    return MAPPING[["model", "parameter_count", "auto_log_count_default_100", "auto_log_fraction_default_100"]]


def k1_matrix() -> pd.DataFrame:
    # Use the last recorded validation point as the run's final point and the
    # best recorded point for the paper-facing K1 comparison.
    rows = []
    for model in sorted(EPOCHS.model.unique()):
        x = EPOCHS[EPOCHS.model == model].sort_values("epoch")
        h = HEALTH[HEALTH.model == model].sort_values("stop_epoch").iloc[-1]
        c = CMA[CMA.model == model]
        cma = float(c.iloc[0].test_kge_median) if len(c) else None
        best = float(x.validation_median_kge.max())
        final = float(x.iloc[-1].validation_median_kge)
        gap = cma - best if cma is not None else None
        rows.append({"model": model, "epochs_recorded": int(x.epoch.max()),
                     "best_epoch": int(x.loc[x.validation_median_kge.idxmax(), "epoch"]),
                     "best_validation_median_kge": best, "final_validation_median_kge": final,
                     "cma_reference_test_median_kge": cma, "cma_minus_best": gap,
                     "abs_gap": abs(gap) if gap is not None else None,
                     "k1_integrity": bool(h.pass_integrity),
                     "k1_no_dead_parameters": bool(h.pass_no_dead_parameters),
                     "k1_no_saturation": bool(h.pass_no_saturation),
                     "k1_convergence_budget": bool(h.pass_convergence_budget),
                     "k1_no_degradation": bool(h.pass_no_degradation)})
    out = pd.DataFrame(rows)
    write(out, "k1_health_matrix.csv")
    bins = pd.DataFrame([{
        "threshold": label,
        "count": int((out.abs_gap <= threshold).sum()) if threshold is not None else int((out.abs_gap > .10).sum()),
    } for label, threshold in [("<=0.02", .02), ("<=0.05", .05), (">0.10", None)]])
    write(bins, "k1_cma_gap_bins.csv")
    return out


def l3() -> pd.DataFrame:
    path = OUT / "l3_auto200/auto100/epochs.csv"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    rows = []
    for model, x in d.groupby("model"):
        old = EPOCHS[EPOCHS.model == model]
        old_best = float(old.validation_median_kge.max())
        y = x.sort_values("epoch")
        new_best = float(y.validation_median_kge.max())
        last_epoch = int(y.epoch.max())
        last_best_epoch = int(y.loc[y.validation_median_kge.idxmax(), "epoch"])
        rows.append({"model": model, "original_recorded_max_epoch": int(old.epoch.max()),
                     "original_best": old_best, "continued_recorded_max_epoch": last_epoch,
                     "continued_best_at_200_budget": new_best, "increment": new_best-old_best,
                     "best_epoch_after_continuation": last_best_epoch,
                     "best_at_end": last_best_epoch == last_epoch,
                     "plateau_or_run_end": last_epoch < 200})
    out = pd.DataFrame(rows).sort_values("model")
    write(out, "l3_plateau_continuation.csv")
    return out


def l4() -> pd.DataFrame:
    path = OUT / "l4_linear100/linear100/epochs.csv"
    if not path.exists():
        return pd.DataFrame()
    d = pd.read_csv(path)
    rows = []
    for model, x in d.groupby("model"):
        z = x.sort_values("epoch")
        rows.append({"model": model, "linear_recorded_max_epoch": int(z.epoch.max()),
                     "linear_best_median_kge": float(z.validation_median_kge.max()),
                     "linear_best_epoch": int(z.loc[z.validation_median_kge.idxmax(), "epoch"]),
                     "linear_final_median_kge": float(z.iloc[-1].validation_median_kge),
                     "linear_plateau_before_100": int(z.epoch.max()) < 100})
    out = pd.DataFrame(rows).sort_values("model")
    write(out, "l4_linear100_results.csv")
    return out


def main() -> None:
    a = l1()
    mapping_table().to_csv(OUT / "l4_mapping_strata.csv", index=False)
    k1_matrix()
    l3(); l4()
    print(a.to_string(index=False))


if __name__ == "__main__":
    main()
