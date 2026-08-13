"""DEPRECATED (2026-08-12): historical dPL checkpoint selection by maximum
VALIDATION median KGE.  Superseded by train-loss-based selection
(scripts/diagnostics/reselect_dpl_trainloss_eval.py) because it used the
1995-2010 validation window that is also the final evaluation window
(selection leakage).  Kept only as provenance of the old rule.

Generate the immutable final evidence tables for round 13."""
from __future__ import annotations
import json
import re
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RUN = ROOT / "results/dpl_round13_20260805"
FINAL = RUN / "final"
FINAL.mkdir(parents=True, exist_ok=True)
MODELS = "alpine1 alpine2 australia collie1 collie2 collie3 flexb flexi flexis gr4j gsfb hbv96 hillslope hymod ihacres modhydrolog mopex1 mopex2 mopex3 mopex4 mopex5 newzealand1 newzealand2 penman plateau simhyd smar susannah1 susannah2 tank tcm topmodel us1 vic wetland xinanjiang".split()
COMPARE = "gr4j mopex1 ihacres collie3 newzealand1 hbv96 flexis tank modhydrolog".split()
CMA = pd.read_csv(ROOT / "results/full300_final_36models_evaluation/full300_kge_model_summary.csv")

def epochs(path):
    return pd.read_csv(path).drop_duplicates(["model", "epoch"], keep="last").sort_values(["model", "epoch"])

def best(d, models):
    rows = []
    for model in models:
        x = d[d.model == model].sort_values("epoch")
        if len(x) != 100 or x.epoch.tolist() != list(range(1, 101)):
            raise RuntimeError(f"{model}: trajectory is not exactly 1..100")
        b = x.sort_values(["validation_median_kge", "epoch"], ascending=[False, True]).iloc[0]
        f = x.iloc[-1]
        rows.append({"model": model, "best_validation_median_kge": float(b.validation_median_kge),
                     "best_epoch": int(b.epoch), "final_epoch": 100,
                     "final_validation_median_kge": float(f.validation_median_kge),
                     "actual_trained_epochs": 100, "actual_epoch_range": "1-100",
                     "final_saturation_ratio": float(f.theta_boundary_fraction),
                     "best_to_final_delta": float(b.validation_median_kge-f.validation_median_kge),
                     "late_best_last5_flag": bool(int(b.epoch) >= 96)})
    return pd.DataFrame(rows)

def old10(model):
    text = (ROOT / f"results/remote_logs_10ep_20260731/dpl_{model}_10ep.log").read_text()
    return float(re.findall(r"Validation KGE Median:\s*([-+]?\d+(?:\.\d+)?)", text)[-1])

def main():
    auto = best(epochs(RUN / "auto100/epochs.csv"), MODELS)
    linear = best(epochs(RUN / "linear100/epochs.csv"), COMPARE)
    h = pd.read_csv(RUN / "auto100/health.csv").drop_duplicates("model", keep="last").set_index("model")
    rows = []
    for _, r in auto.iterrows():
        hh = h.loc[r.model]; c = CMA[CMA.model == r.model]
        cma = float(c.iloc[0].test_kge_median) if len(c) else None
        z = str(hh.get("permanently_zero_parameters", ""))
        zcount = 0 if z in {"", "nan"} else len([x for x in z.split(";") if x])
        train_nf = int(hh.train_nonfinite_prediction_count); val_nf = int(hh.validation_nonfinite_prediction_count)
        hard = bool(train_nf or val_nf or zcount or not hh.pass_integrity)
        rows.append({**r.to_dict(), "train_nonfinite_count": train_nf, "validation_nonfinite_count": val_nf,
                     "permanent_zero_gradient_parameter_count": zcount, "cma_reference_test_median_kge": cma,
                     "absolute_cma_gap": abs(cma-r.best_validation_median_kge) if cma is not None else None,
                     "integrity_verdict": "HARD_FAILURE" if hard else "PASS",
                     "hard_failure_nonfinite": bool(train_nf or val_nf), "hard_failure_permanent_zero_gradient": bool(zcount),
                     "diagnostic_high_saturation_flag": bool(r.final_saturation_ratio >= .20),
                     "diagnostic_late_best_flag": bool(r.late_best_last5_flag),
                     "diagnostic_degradation_flag": bool(r.best_to_final_delta > .05)})
    af = pd.DataFrame(rows)
    af.to_csv(FINAL / "auto100_final_36models.csv", index=False, float_format="%.8f")
    af[["model", "integrity_verdict", "hard_failure_nonfinite", "hard_failure_permanent_zero_gradient",
        "diagnostic_high_saturation_flag", "diagnostic_late_best_flag", "diagnostic_degradation_flag"]].to_csv(FINAL / "health_final_36models.csv", index=False)

    ai = af.set_index("model"); li = linear.set_index("model"); ar = []
    for model in COMPARE:
        x, y = ai.loc[model], li.loc[model]; l10 = old10(model)
        ar.append({"model": model, "linear10": l10, "linear100_best": float(y.best_validation_median_kge),
                   "linear100_best_epoch": int(y.best_epoch), "linear100_final": float(y.final_validation_median_kge),
                   "linear100_actual_epochs": "1-100", "auto100_best": float(x.best_validation_median_kge),
                   "auto100_best_epoch": int(x.best_epoch), "auto100_final": float(x.final_validation_median_kge),
                   "auto100_actual_epochs": "1-100", "epoch_net_effect": float(y.best_validation_median_kge-l10),
                   "mapping_net_effect": float(x.best_validation_median_kge-y.best_validation_median_kge)})
    attr = pd.DataFrame(ar)
    attr.to_csv(FINAL / "mapping_attribution_9models.csv", index=False, float_format="%.8f")
    linear.to_csv(FINAL / "linear100_9models.csv", index=False, float_format="%.8f")

    gr = []
    for _, x in af.iterrows():
        l10 = old10(x.model)
        gr.append({"model": x.model, "linear10_old": l10, "auto100_best": x.best_validation_median_kge,
                   "improvement_vs_linear10": x.best_validation_median_kge-l10,
                   "cma_reference": x.cma_reference_test_median_kge, "absolute_cma_gap": x.absolute_cma_gap,
                   "best_epoch": x.best_epoch})
    gd = pd.DataFrame(gr)
    gd.to_csv(FINAL / "cma_gap_final.csv", index=False, float_format="%.8f")
    pd.DataFrame([{"bin": "<=0.02", "count": int((gd.absolute_cma_gap <= .02).sum())},
                  {"bin": "<=0.05", "count": int((gd.absolute_cma_gap <= .05).sum())},
                  {"bin": ">0.10", "count": int((gd.absolute_cma_gap > .10).sum())}]).to_csv(FINAL / "cma_gap_bins_final.csv", index=False)

    trace = json.loads((FINAL / "simhyd_failure_trace.json").read_text())
    pd.DataFrame([{"epoch": trace.get("epoch"), "batch_step": trace.get("batch_step"), "source": trace.get("source"),
                   "loss": trace.get("loss"), "gradient_norm_before_clip": trace.get("gradient_norm_before_clip"),
                   "theta_gradient_finite_count": trace.get("theta_gradient_stats", {}).get("finite_count"),
                   "theta_gradient_total_count": trace.get("theta_gradient_stats", {}).get("total_count"),
                   "network_finite_after_step": trace.get("network_stats_after_step", {}).get("finite"),
                   "bad_theta_entries": json.dumps(trace.get("theta_bad_entries", []))}]).to_csv(FINAL / "simhyd_failure_trace.csv", index=False)
    vic_trace = json.loads((FINAL / "vic_failure_trace.json").read_text())
    pd.DataFrame([{"epoch": vic_trace.get("epoch"), "batch_step": vic_trace.get("batch_step"), "source": vic_trace.get("source"),
                   "loss": vic_trace.get("loss"), "gradient_finite_count": vic_trace.get("gradient_stats", {}).get("finite_count"),
                   "gradient_total_count": vic_trace.get("gradient_stats", {}).get("total_count"),
                   "theta_gradient_finite_count": vic_trace.get("theta_gradient_stats", {}).get("finite_count"),
                   "theta_gradient_total_count": vic_trace.get("theta_gradient_stats", {}).get("total_count")}]).to_csv(FINAL / "vic_failure_trace.csv", index=False)

    hard = af.loc[af.integrity_verdict == "HARD_FAILURE", "model"].tolist()
    lag = gd.sort_values("absolute_cma_gap", ascending=False).head(8)[["model", "absolute_cma_gap"]].to_string(index=False)
    bins = [("<=0.02", int((gd.absolute_cma_gap <= .02).sum())),
            ("<=0.05", int((gd.absolute_cma_gap <= .05).sum())),
            (">0.10", int((gd.absolute_cma_gap > .10).sum()))]
    sat_models = af.loc[af.diagnostic_high_saturation_flag, "model"].tolist()
    late_models = af.loc[af.diagnostic_late_best_flag, "model"].tolist()
    degraded_models = af.loc[af.diagnostic_degradation_flag, "model"].tolist()
    lines = ["# dPL Round 13 Final Report", "",
             "## One-line verdict", "",
             "The fixed-budget training and paired mapping experiments are complete; the pipeline has two reproducible backward hard failures (SIMHYD and VIC), so the diagnostic phase is not fully healthy and their final KGE values are not valid final-accuracy claims.", "",
             "## Fixed-budget contract", "",
             "Auto: 36/36 models, exactly epochs 1-100, plateau early stopping disabled. Linear control: nine paired models, exactly epochs 1-100. Best point is maximum validation median KGE, with earliest epoch on ties. The two arms therefore have comparable fixed budgets.", "",
             "## Auto health", "",
             f"Integrity: {int((af.integrity_verdict == 'PASS').sum())}/36 PASS; hard failures: {', '.join(hard) if hard else 'none'}. Permanent-zero-gradient: {int((af.permanent_zero_gradient_parameter_count > 0).sum())}/36. Diagnostic flags: high saturation {len(sat_models)}/36 ({', '.join(sat_models)}); best in last five epochs {len(late_models)}/36 ({', '.join(late_models)}); best-to-final decline >0.05 {len(degraded_models)}/36 ({', '.join(degraded_models)}). Saturation is not a hard failure.", "",
             "All 36 auto trajectories and all 9 linear trajectories contain exactly epochs 1-100. The complete machine-readable per-model matrix is `auto100_final_36models.csv`; SIMHYD and VIC are retained in the table for traceability but must be excluded from claims requiring valid final predictions.", "",
             "## SIMHYD", "",
             "The first forward non-finite is exposed at epoch 64, training batch 103; the preceding batch 102 has finite q and loss but 4/700 theta gradients are NaN. `clip_grad_norm_` is NaN, the optimizer step corrupts all 77,831 network values, and the next batch exposes seven NaN physical parameters. Autograd reports `DivBackward0` at `dmotpy/models/core/simhyd.py:84`, `soil / smsc_safe`. This is backward gradient contamination, not a forward state overflow. No formula-preserving fix was justified; no formula, loss, interface, or final failure evidence was changed.", "",
             "## VIC", "",
             "The first failure is epoch 57, training batch 145. q and loss are finite; 997/1000 theta gradients and 1,799/78,602 network-gradient entries are finite. This is an independent backward numerical failure. No VIC formula patch was applied, and the final KGE is not treated as valid accuracy.", "",
             "## TOPMODEL", "",
             "The perturbation uses the reproducible epoch-100 checkpoint because the scalar best epoch 88 checkpoint was not saved. Current f has median 0.005118 and minimum 0.000655 on the normalized [0,1] interval; 425/531 basins are near the lower boundary. KGE is 0.468881 at current f, 0.473762 at 0.9x, 0.504945 at 0.5x, 0.231557 at 0.1x, -6.142345 when each basin is reduced to 1% of its current f, and -147.203461 at exact zero. q remains finite in every case. Classification: **near-boundary numerical cliff**. A 10% local perturbation is stable, but the learned solution is not robust across the lower-boundary neighborhood. The relevant expression is `baseflow_4(q0, f, S2) = q0 * exp(-f * S2)` at `dmotpy/models/flux/baseflow.py:38-45`, called from `dmotpy/models/core/topmodel.py:131` and `:155`; f=0 removes the decay and produces very large finite baseflow. This is a model-boundary limitation/diagnostic issue, not evidence of NaN/Inf or a justified formula change.", "",
             "## Mapping attribution", "",
             f"Positive mapping effects: {int((attr.mapping_net_effect > 0).sum())}/9; negative: {int((attr.mapping_net_effect < 0).sum())}/9. All nine epoch effects are positive. Collie3's +0.359744 is predominantly an epoch effect (+0.014502 mapping); NewZealand1 retains a +0.095562 mapping effect; HBV96 has a small -0.006633 mapping effect despite a strong epoch effect. Flexis has +0.140354 epoch effect and -0.009922 mapping effect, with its auto best-to-final decline requiring best-checkpoint selection. Overall claim strength: **partial, model-dependent support**, not a universal or sole-cause claim.", "", attr.to_string(index=False), "",
             "## CMA gap", "", "Final auto best-point absolute gap bins: " + "; ".join(f"{name}: {count}/36" for name, count in bins) + ".", "", "Largest gaps:", "", lag, "",
             "## Files and change record", "", "This round added/updated only experiment evidence and diagnostic reports under `results/dpl_round13_20260805/final/`, plus the runner's contract serialization in `scripts/diagnostics/round13_m1.py`. No public interface, model equation, loss, MLP architecture, forcing contract, or existing raw checkpoint was changed.", "", "Original failure traces remain in `simhyd_failure_trace.json`, `vic_failure_trace.json`, and the corresponding CSV summaries. Final model tables are not a silent replacement for those failures."]
    (FINAL / "round13_final_report.md").write_text("\n".join(lines) + "\n")
    (FINAL / "simhyd_diagnostic_report.md").write_text("""# SIMHYD diagnostic

The first forward non-finite was exposed at epoch 64, training batch 103. The preceding batch 102 had finite q and loss (`0.5724595785`), but 4/700 theta-gradient entries were NaN. The gradient clip norm was NaN; the optimizer step then made all 77,831 network values non-finite, and the next batch exposed all seven physical parameters as NaN. The first bad basin recorded in the failing update was `11532500`; the first basin exposing the corrupted forward was `1639500`.

Autograd anomaly detection identifies `DivBackward0` at `dmotpy/models/core/simhyd.py:84`, `soil / smsc_safe`. Thus the contamination occurs in backward, after a finite forward, and propagates as `gradient -> optimizer state/MLP -> physical parameters -> next forward`. The original failure is preserved in `simhyd_failure_trace.json` and `.csv`. No formula, loss, interface, clamp, or `nan_to_num` patch was applied because a formula-preserving fix was not established.
""")
    (FINAL / "vic_diagnostic_report.md").write_text("""# VIC diagnostic

The first failure was epoch 57, training batch 145. q and loss were finite (`0.6782485247`), while 997/1000 theta-gradient entries and 1,799/78,602 network-gradient entries were finite. This is an independent backward gradient contamination, not a forward NaN/Inf. The original trace is preserved in `vic_failure_trace.json` and `.csv`; no VIC equation or formula-preserving numerical patch was applied.
""")
    (FINAL / "topmodel_boundary_report.md").write_text("""# TOPMODEL boundary report

Perturbations are in `topmodel_local_boundary_sensitivity.csv`. The scalar best is epoch 88 but no epoch-88 checkpoint exists, so the reproducible epoch-100 checkpoint is used transparently. Current f has median 0.005118 and minimum 0.000655 on [0,1], with 425/531 basins near the lower boundary. KGE is 0.468881 at current f, 0.473762 at 0.9x, 0.504945 at 0.5x, 0.231557 at 0.1x, -6.142345 when each basin is reduced to 1% of its current f, and -147.203461 at exact zero. q remains finite for all perturbations. The replay also records finite S1/S2 state ranges in the CSV.

Classification: **near-boundary numerical cliff**. The current solution is locally stable to a 10% decrease but not robust across the lower-boundary neighborhood. The relevant expression is `q0 * exp(-f * S2)` in `dmotpy/models/flux/baseflow.py:38-45`, called from `dmotpy/models/core/topmodel.py:131` and `:155`. Exact f=0 yields extreme finite output by removing the decay; this is a boundary-sensitivity limitation, not NaN/Inf evidence. No formula change is justified in this round.
""")
    print("finalized", FINAL)

if __name__ == "__main__":
    main()
