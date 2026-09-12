#!/usr/bin/env python3
"""Frozen inputs and ranking helpers for deterministic OOB model selection."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
BENCHMARK = REPO / "project/benchmark"
SEEN = BENCHMARK / "results/seenbasin_remaining_analysis_20260901"
OUT = BENCHMARK / "results/oob_model_selection_20260901"
MODELS = list(pd.read_csv(SEEN / "SEENBASIN_MASTER_MODEL_SUMMARY.csv").model)
SELECTION_RAW = ["G", "R", "D", "U", "K_joint", "P", "A"]
SELECTION_RANK = [f"{x}_pct" for x in SELECTION_RAW]
CONTRAST_RAW = ["D", "U", "A", "P", "K_joint"]


def rank01(values: pd.Series) -> pd.Series:
    n = len(values)
    if n <= 1:
        return pd.Series(0.5, index=values.index)
    return (values.rank(method="average") - 1.0) / (n - 1.0)


def euclidean(frame: pd.DataFrame, columns: list[str], center: np.ndarray) -> pd.Series:
    x = frame[columns].to_numpy(float)
    return pd.Series(np.sqrt(np.square(x - center).sum(axis=1)), index=frame.index)


def load_features() -> tuple[pd.DataFrame, dict[str, float]]:
    master = pd.read_csv(SEEN / "SEENBASIN_MASTER_MODEL_SUMMARY.csv")
    b01 = pd.read_csv(SEEN / "agent_B/B01_MODEL_ATTRIBUTE_GAP_ASSOCIATION.csv")
    required = {"model", "G_seen_median", "reproducibility_median", "D_theta_median", "restart_mean_u_sd", "ic_median", "dpl_median", "parameter_count"}
    missing = required - set(master.columns)
    if missing:
        raise RuntimeError(f"master model summary missing columns: {sorted(missing)}")
    signal = b01.assign(abs_rho=b01.rho.abs())
    signal_summary = signal.groupby("model", sort=True).agg(
        A=("q_value", lambda x: int((x < 0.05).sum())),
        max_abs_rho_G_attribute=("abs_rho", "max"),
        median_abs_rho_G_attribute=("abs_rho", "median"),
        significant_attribute_count=("q_value", lambda x: int((x < 0.05).sum())),
    ).reset_index()
    frame = master[["model", "G_seen_median", "reproducibility_median", "D_theta_median", "restart_mean_u_sd", "ic_median", "dpl_median", "parameter_count"]].copy()
    frame = frame.rename(columns={"G_seen_median": "G", "reproducibility_median": "R", "D_theta_median": "D", "restart_mean_u_sd": "U", "parameter_count": "P", "ic_median": "K_IC", "dpl_median": "K_dPL"})
    frame["K_joint"] = frame[["K_IC", "K_dPL"]].min(axis=1)
    frame = frame.merge(signal_summary, on="model", validate="one_to_one")
    if len(frame) != 36 or set(frame.model) != set(MODELS):
        raise RuntimeError("candidate pool is not exactly the frozen 36-model set")
    thresholds = {
        "G_median": float(frame.G.median()),
        "R_median": float(frame.R.median()),
        "K_joint_Q25": float(frame.K_joint.quantile(0.25)),
        "G_Q33": float(frame.G.quantile(1 / 3)),
        "G_Q67": float(frame.G.quantile(2 / 3)),
        "R_Q33": float(frame.R.quantile(1 / 3)),
        "R_Q67": float(frame.R.quantile(2 / 3)),
        "A_Q33": float(frame.A.quantile(1 / 3)),
        "A_Q67": float(frame.A.quantile(2 / 3)),
        "P_Q33": float(frame.P.quantile(1 / 3)),
        "P_Q67": float(frame.P.quantile(2 / 3)),
    }
    for column in SELECTION_RAW:
        frame[f"{column}_pct"] = rank01(frame[column])
    frame["G_class"] = np.where(frame.G <= thresholds["G_median"], "low", "high")
    frame["R_class"] = np.where(frame.R <= thresholds["R_median"], "low", "high")
    frame["quadrant"] = np.select(
        [((frame.G_class == "low") & (frame.R_class == "high")), ((frame.G_class == "low") & (frame.R_class == "low")), ((frame.G_class == "high") & (frame.R_class == "high")), ((frame.G_class == "high") & (frame.R_class == "low"))],
        ["Q1_LOW_G_HIGH_R", "Q2_LOW_G_LOW_R", "Q3_HIGH_G_HIGH_R", "Q4_HIGH_G_LOW_R"], default="UNDEFINED")
    frame["A_tercile"] = np.select([frame.A <= thresholds["A_Q33"], frame.A >= thresholds["A_Q67"]], ["low", "high"], default="medium")
    frame["P_tercile"] = np.select([frame.P <= thresholds["P_Q33"], frame.P >= thresholds["P_Q67"]], ["low", "high"], default="medium")
    frame["G_extreme"] = np.select([frame.G <= thresholds["G_Q33"], frame.G >= thresholds["G_Q67"]], ["low", "high"], default="middle")
    frame["R_extreme"] = np.select([frame.R <= thresholds["R_Q33"], frame.R >= thresholds["R_Q67"]], ["low", "high"], default="middle")
    frame["performance_gate_pass"] = frame.K_joint >= thresholds["K_joint_Q25"]
    return frame, thresholds


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, float_format="%.10f")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
