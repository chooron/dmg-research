"""Round-13 offline tables after the two forced-100-epoch arms finish."""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results/dpl_round13_20260805"
MODELS = [
    "alpine1", "alpine2", "australia", "collie1", "collie2", "collie3",
    "flexb", "flexi", "flexis", "gr4j", "gsfb", "hbv96", "hillslope",
    "hymod", "ihacres", "modhydrolog", "mopex1", "mopex2", "mopex3",
    "mopex4", "mopex5", "newzealand1", "newzealand2", "penman", "plateau",
    "simhyd", "smar", "susannah1", "susannah2", "tank", "tcm", "topmodel",
    "us1", "vic", "wetland", "xinanjiang",
]
COMPARE = ["gr4j", "mopex1", "ihacres", "collie3", "newzealand1", "hbv96", "flexis", "tank", "modhydrolog"]

CMA = pd.read_csv(ROOT / "results/full300_final_36models_evaluation/full300_kge_model_summary.csv")
MAPPING = pd.read_csv(ROOT / "results/dpl_training_pilot_20260801/i2_mapping/i2_model_summary.csv")


def dedup(path: Path, keys: list[str]) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d.drop_duplicates(keys, keep="last").sort_values(keys).reset_index(drop=True)
    d.to_csv(path, index=False)
    return d


def arm_epochs(arm: str) -> pd.DataFrame:
    return dedup(OUT / arm / "epochs.csv", ["model", "epoch"])


def best_rows(d: pd.DataFrame, models: list[str] | None = None) -> pd.DataFrame:
    models = MODELS if models is None else models
    rows = []
    for model in models:
        x = d[d.model == model].sort_values("epoch")
        if len(x) != 100 or x.epoch.tolist() != list(range(1, 101)):
            raise RuntimeError(f"{model}: expected epochs 1..100, got {x.epoch.min()}..{x.epoch.max()} ({len(x)})")
        # Stable best-point rule: maximum validation median, earliest epoch on ties.
        row = x.sort_values(["validation_median_kge", "epoch"], ascending=[False, True]).iloc[0]
        first = x.iloc[0]
        final = x.iloc[-1]
        rows.append({
            "model": model, "actual_epoch_range": "1-100", "actual_epoch_count": 100,
            "best_epoch": int(row.epoch), "best_validation_median_kge": float(row.validation_median_kge),
            "epoch1_validation_median_kge": float(first.validation_median_kge),
            "final_validation_median_kge": float(final.validation_median_kge),
            "absolute_improvement": float(row.validation_median_kge - first.validation_median_kge),
            "best_minus_final": float(row.validation_median_kge - final.validation_median_kge),
            "final_boundary_fraction": float(final.theta_boundary_fraction),
            "best_at_end": bool(int(row.epoch) == 100),
        })
    return pd.DataFrame(rows)


def health(auto: pd.DataFrame) -> pd.DataFrame:
    h = pd.read_csv(OUT / "auto100/health.csv").drop_duplicates("model", keep="last")
    rows = []
    for _, r in auto.iterrows():
        model = r.model
        hr = h[h.model == model].iloc[-1]
        cm = CMA[CMA.model == model]
        cma = float(cm.iloc[0].test_kge_median) if len(cm) else None
        gap = abs(cma - r.best_validation_median_kge) if cma is not None else None
        # K1 learning uses the L1 replacement rule, not best - epoch 1 > .05.
        effective = (gap <= .05) if gap is not None else None
        rows.append({
            "model": model, "actual_epoch_range": r.actual_epoch_range,
            "best_epoch": int(r.best_epoch), "best_validation_median_kge": r.best_validation_median_kge,
            "cma_reference_test_median_kge": cma, "abs_gap": gap,
            "k1_integrity": bool(hr.pass_integrity), "k1_effective_learning": effective,
            "k1_no_dead_parameters": bool(hr.pass_no_dead_parameters),
            "k1_no_saturation": bool(r.final_boundary_fraction < .20),
            "k1_convergence_budget": bool(r.best_epoch < 96),
            "k1_no_degradation": bool(r.best_minus_final <= .05),
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "k1_health_matrix.csv", index=False, float_format="%.8f")
    bins = pd.DataFrame([
        {"threshold": "<=0.02", "count": int((out.abs_gap <= .02).sum())},
        {"threshold": "<=0.05", "count": int((out.abs_gap <= .05).sum())},
        {"threshold": ">0.10", "count": int((out.abs_gap > .10).sum())},
    ])
    bins.to_csv(OUT / "k1_cma_gap_bins.csv", index=False)
    return out


def old_linear10() -> dict[str, float]:
    out = {}
    for model in COMPARE:
        path = ROOT / f"results/remote_logs_10ep_20260731/dpl_{model}_10ep.log"
        text = path.read_text()
        values = re.findall(r"Validation KGE Median:\s*([-+]?\d+(?:\.\d+)?)", text)
        if not values:
            raise RuntimeError(f"no validation median in {path}")
        out[model] = float(values[-1])
    return out


def attribution(auto: pd.DataFrame, linear: pd.DataFrame) -> pd.DataFrame:
    old = old_linear10()
    a = auto.set_index("model")
    l = linear.set_index("model")
    rows = []
    for model in COMPARE:
        lin = float(l.loc[model, "best_validation_median_kge"])
        aut = float(a.loc[model, "best_validation_median_kge"])
        old10 = old[model]
        rows.append({
            "model": model, "linear_plus_10ep_old": old10,
            "linear_plus_uniform_best": lin, "linear_uniform_best_epoch": int(l.loc[model, "best_epoch"]),
            "linear_uniform_actual_epochs": "1-100", "auto_plus_uniform_best": aut,
            "auto_uniform_best_epoch": int(a.loc[model, "best_epoch"]), "auto_uniform_actual_epochs": "1-100",
            "epoch_net_effect": lin - old10, "mapping_net_effect": aut - lin,
        })
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "attribution_table.csv", index=False, float_format="%.8f")
    return out


def global_table(auto: pd.DataFrame) -> pd.DataFrame:
    old = old_linear10()
    rows = []
    for _, r in auto.iterrows():
        model = r.model
        path = ROOT / f"results/remote_logs_10ep_20260731/dpl_{model}_10ep.log"
        values = re.findall(r"Validation KGE Median:\s*([-+]?\d+(?:\.\d+)?)", path.read_text())
        old10 = float(values[-1]) if values else None
        cm = CMA[CMA.model == model]
        cma = float(cm.iloc[0].test_kge_median) if len(cm) else None
        rows.append({"model": model, "old_10ep": old10, "new_auto_uniform_best": r.best_validation_median_kge,
                     "improvement": r.best_validation_median_kge - old10 if old10 is not None else None,
                     "cma_reference": cma, "new_abs_gap_to_cma": abs(cma-r.best_validation_median_kge) if cma is not None else None,
                     "new_best_epoch": int(r.best_epoch), "actual_epochs": "1-100"})
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "global_comparison.csv", index=False, float_format="%.8f")
    return out


def main() -> None:
    auto = best_rows(arm_epochs("auto100"))
    linear = best_rows(arm_epochs("linear100"), COMPARE)
    auto.to_csv(OUT / "auto_uniform_best.csv", index=False, float_format="%.8f")
    linear.to_csv(OUT / "linear_uniform_best.csv", index=False, float_format="%.8f")
    health(auto)
    attribution(auto, linear)
    global_table(auto)
    MAPPING.to_csv(OUT / "mapping_strata.csv", index=False)
    print("K1 gap bins:")
    print(pd.read_csv(OUT / "k1_cma_gap_bins.csv").to_string(index=False))
    print("\nAttribution:")
    print(pd.read_csv(OUT / "attribution_table.csv").to_string(index=False))


if __name__ == "__main__":
    main()
