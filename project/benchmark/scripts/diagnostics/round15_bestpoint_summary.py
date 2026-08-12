"""Round-15 unified validation-best checkpoint summary (read-only)."""
from __future__ import annotations

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/dpl_round13_20260805"
OUT = ROOT / "results/dpl_round15_20260808"
SAFE_LAST_EPOCH = {"simhyd": 63, "vic": 56}


def select_best(frame: pd.DataFrame, metric: str) -> pd.Series:
    usable = frame[frame.epoch <= SAFE_LAST_EPOCH.get(frame.model.iloc[0], 100)]
    return usable.sort_values([metric, "epoch"], ascending=[False, True]).iloc[0]


def bins(series: pd.Series) -> dict[str, int]:
    return {"<=0.02": int((series <= .02).sum()), "<=0.05": int((series <= .05).sum()), ">0.10": int((series > .10).sum())}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    epochs = pd.read_csv(RUN / "auto100/epochs.csv").drop_duplicates(["model", "epoch"], keep="last")
    health = pd.read_csv(RUN / "auto100/health.csv").drop_duplicates("model", keep="last").set_index("model")
    cma_basin = pd.read_csv(ROOT / "results/full300_final_36models_evaluation/full300_kge_by_basin.csv")
    cma = cma_basin.groupby("model", as_index=False).agg(cma_test_median=("test_kge", "median"), cma_test_mean=("test_kge", "mean"))
    rows = []
    for model, frame in epochs.groupby("model", sort=True):
        med = select_best(frame, "validation_median_kge")
        mean = select_best(frame, "validation_mean_kge")
        ref = cma[cma.model == model].iloc[0]
        failed = not bool(health.loc[model, "pass_integrity"])
        rows.append({
            "model": model,
            "integrity_hard_failure": failed,
            "pre_failure_checkpoint_only": model in SAFE_LAST_EPOCH,
            "median_best_epoch": int(med.epoch),
            "median_at_median_best": float(med.validation_median_kge),
            "mean_at_median_best": float(med.validation_mean_kge),
            "mean_best_epoch": int(mean.epoch),
            "mean_at_mean_best": float(mean.validation_mean_kge),
            "median_at_mean_best": float(mean.validation_median_kge),
            "cma_test_median": float(ref.cma_test_median),
            "cma_test_mean": float(ref.cma_test_mean),
        })
    result = pd.DataFrame(rows)
    result["median_abs_gap"] = (result.median_at_median_best - result.cma_test_median).abs()
    result["mean_abs_gap"] = (result.mean_at_mean_best - result.cma_test_mean).abs()
    result["dpl_ge_cma_median"] = result.median_at_median_best >= result.cma_test_median
    result["dpl_ge_cma_mean"] = result.mean_at_mean_best >= result.cma_test_mean
    result.to_csv(OUT / "bestpoint_cma_comparison_36models.csv", index=False, float_format="%.8f")
    summaries = []
    for label, subset in (("all36_pre_failure_for_simhyd_vic", result), ("healthy34_only", result[~result.integrity_hard_failure])):
        for metric, column in (("median", "median_abs_gap"), ("mean", "mean_abs_gap")):
            summaries.append({"population": label, "metric": metric, **bins(subset[column]), "dpl_ge_cma": int(subset[f"dpl_ge_cma_{metric}"].sum()), "n_models": len(subset)})
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "bestpoint_cma_gap_bins.csv", index=False)
    lines = ["# Round 15 P2: Unified best-checkpoint statistics", "",
             "Rule: for each metric separately, select the earliest epoch attaining the maximum full-validation value over the valid training trajectory. Median tables select by validation median KGE; mean tables select by validation mean KGE. SIMHYD is restricted to epochs 1-63 and VIC to 1-56, before their first contaminated optimizer update. All other models use epochs 1-100.", "",
             "The 36-model rows retain SIMHYD/VIC only as pre-failure evidence. The healthy-34 rows exclude them.", "", "## Gap bins", "", summary.to_string(index=False), "",
             "## Comparison to old linear-10 baseline", "", "Old median-gap counts were <=0.02: 18/36, <=0.05: 25/36, >0.10: 4/36. Mean equivalents cannot be reconstructed from the archived 10-epoch summary because it retained only median KGE.", "",
             "## DPL >= CMA", "", result[["model", "dpl_ge_cma_median", "dpl_ge_cma_mean", "integrity_hard_failure", "pre_failure_checkpoint_only"]].to_string(index=False)]
    (OUT / "p2_bestpoint_report.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
