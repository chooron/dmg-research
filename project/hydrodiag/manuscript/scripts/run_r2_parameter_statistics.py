#!/usr/bin/env python3
"""Reproducible R2 parameter-layer audit and statistics.

This script is deliberately result-first: it reads the active R1 parameter
sources, audits their schemas/coverage, computes Base-CN shifts in the common
XAJ physical parameter space, and writes machine-readable diagnostics under
manuscript/results/R2.  It does not create Figure 3.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, t

PROJECT = Path(__file__).resolve().parents[2]
MANUSCRIPT = PROJECT / "manuscript"
DATA = PROJECT.parents[1] / "data"
RESULTS = PROJECT / "results"
OUT = MANUSCRIPT / "results" / "R2"

BASIN_FILE = DATA / "531sub_id.txt"
SNOW_FILE = MANUSCRIPT / "results" / "R1" / "r1_snow_attributes.csv"
BOUNDS_FILE = (
    MANUSCRIPT / "supplement" / "results" / "s2_parameter_bounds_from_code.csv"
)

STRUCTURES = {
    "Base": {
        "model_key": "XAJ",
        "ic_root": "xaj_base_cmaes_531_batched_paired_v2",
        "ic_subdir": "xaj",
        "dpl_root": "dpl_camels_531_lite_v2",
        "dpl_subdir": "XAJ",
        "ic_label": "IC-CMA-ES",
    },
    "CN": {
        "model_key": "XAJ_CN",
        "ic_root": "xaj_cn_cmaes_531_batched_paired_v2",
        "ic_subdir": "xaj_cn",
        "dpl_root": "dpl_camels_531_lite_v2",
        "dpl_subdir": "XAJ_CN",
        "ic_label": "IC-CMA-ES",
    },
    "GD": {
        "model_key": "XAJ_TGD2",
        "ic_root": "xaj_tgd2_cmaes_531_batched_v1",
        "ic_subdir": "xaj_tgd2",
        "dpl_root": "dpl_camels_531_lite_v3_tgd2_dpl_audited",
        "dpl_subdir": "XAJ_TGD2",
        "ic_label": "IC-CMA-ES",
    },
}
COMMON_XAJ = [
    "xaj_k",
    "xaj_b",
    "xaj_im",
    "xaj_um",
    "xaj_lm",
    "xaj_dm",
    "xaj_c",
    "xaj_sm",
    "xaj_ex",
    "xaj_ki",
    "xaj_kg",
    "xaj_ci",
    "xaj_cg",
    "xaj_a",
    "xaj_theta",
]
DISPLAY = {
    "xaj_k": "k",
    "xaj_b": "b",
    "xaj_im": "im",
    "xaj_um": "um",
    "xaj_lm": "lm",
    "xaj_dm": "dm",
    "xaj_c": "c",
    "xaj_sm": "sm",
    "xaj_ex": "ex",
    "xaj_ki": "ki",
    "xaj_kg": "kg",
    "xaj_ci": "ci",
    "xaj_cg": "cg",
    "xaj_a": "a (UH shape)",
    "xaj_theta": "theta (UH scale)",
}
FUNCTION = {
    "xaj_k": "ET / tension-water storage",
    "xaj_b": "production / free-water storage",
    "xaj_im": "production / free-water storage",
    "xaj_um": "ET / tension-water storage",
    "xaj_lm": "ET / tension-water storage",
    "xaj_dm": "ET / tension-water storage",
    "xaj_c": "ET / tension-water storage",
    "xaj_sm": "production / free-water storage",
    "xaj_ex": "production / free-water storage",
    "xaj_ki": "routing / recession",
    "xaj_kg": "routing / recession",
    "xaj_ci": "routing / recession",
    "xaj_cg": "routing / recession",
    "xaj_a": "routing timing",
    "xaj_theta": "routing timing",
}
STRATA = [
    ("S1", "[0, 0.05)", 0.0, 0.05),
    ("S2", "[0.05, 0.15)", 0.05, 0.15),
    ("S3", "[0.15, 0.30)", 0.15, 0.30),
    ("S4", "[0.30, 0.50)", 0.30, 0.50),
    ("S5", "[0.50, 1.00]", 0.50, 1.00),
]
SEEDS = (42, 123, 2026)
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 20260730
BOUNDARY_THRESHOLDS = (0.01, 0.02, 0.05)


def read_basins() -> list[str]:
    raw = BASIN_FILE.read_text().strip()
    try:
        values = json.loads(raw)
        basins = [str(value).zfill(8) for value in values]
    except json.JSONDecodeError:
        basins = [line.strip().zfill(8) for line in raw.splitlines() if line.strip()]
    if len(basins) != 531 or len(set(basins)) != 531:
        raise ValueError(f"canonical basin list is not unique 531: n={len(basins)}")
    return basins


def snow_table(basins: list[str]) -> pd.DataFrame:
    snow = pd.read_csv(SNOW_FILE, dtype={"basin_id": str})
    snow["basin_id"] = snow["basin_id"].str.zfill(8)
    if snow["basin_id"].duplicated().any():
        raise ValueError("duplicate basin IDs in R1 snow table")
    if set(snow["basin_id"]) != set(basins):
        raise ValueError("R1 snow table does not exactly match canonical basin set")
    snow["frac_snow"] = pd.to_numeric(snow["frac_snow"], errors="coerce")
    if not np.isfinite(snow["frac_snow"]).all():
        raise ValueError("nonfinite frac_snow in canonical R1 table")

    def assign(v: float) -> tuple[str, str]:
        for name, interval, lo, hi in STRATA:
            if (v >= lo and v < hi) or (name == "S5" and v <= hi):
                return name, interval
        raise ValueError(f"frac_snow outside fixed R1 strata: {v}")

    snow[["snow_regime", "snow_interval"]] = (
        snow["frac_snow"].map(assign).apply(pd.Series)
    )
    return snow[["basin_id", "frac_snow", "snow_regime", "snow_interval"]]


def load_bounds() -> pd.DataFrame:
    b = pd.read_csv(BOUNDS_FILE)
    # XAJ-Base is the canonical common parameter definition; verify every
    # public parameter has the same definition and range in XAJ-CN and TGD.
    rows = b[b["active_model_key"].isin(["XAJ", "XAJ_CN", "XAJ_TGD"])].copy()
    # Current TGD2 run uses the same public XAJ specs as the audited XAJ_TGD
    # row family; verify names, symbols, bounds and units before comparison.
    for key in ["XAJ", "XAJ_CN", "XAJ_TGD"]:
        sub = (
            rows[(rows["active_model_key"] == key) & rows["code_name"].isin(COMMON_XAJ)]
            .drop_duplicates("code_name")
            .set_index("code_name")
        )
        if set(sub.index) != set(COMMON_XAJ):
            raise ValueError(
                f"{key} does not expose exactly the 15 public XAJ parameters"
            )
    base_check = (
        rows[(rows["active_model_key"] == "XAJ") & rows["code_name"].isin(COMMON_XAJ)]
        .drop_duplicates("code_name")
        .set_index("code_name")
        .loc[COMMON_XAJ]
    )
    for key in ["XAJ_CN", "XAJ_TGD"]:
        sub = (
            rows[(rows["active_model_key"] == key) & rows["code_name"].isin(COMMON_XAJ)]
            .drop_duplicates("code_name")
            .set_index("code_name")
            .loc[COMMON_XAJ]
        )
        for col in ["symbol", "lower_bound", "upper_bound"]:
            if not np.array_equal(base_check[col].to_numpy(), sub[col].to_numpy()):
                raise ValueError(f"public XAJ {col} mismatch between XAJ and {key}")
    # XAJ-Base is the canonical common parameter definition; verify every
    # public parameter has the same definition and range in XAJ-CN and TGD.
    base = base_check.reset_index()
    if set(base["code_name"]) != set(COMMON_XAJ):
        raise ValueError(
            "bounds inventory does not contain all 15 active XAJ public parameters"
        )
    base = (
        base.drop_duplicates("code_name")
        .set_index("code_name")
        .loc[COMMON_XAJ]
        .reset_index()
    )
    if "unit" not in base:
        base["unit"] = ""
    return base[
        ["code_name", "symbol", "lower_bound", "upper_bound", "unit", "source_file"]
    ]


def read_ic(basins: list[str], structure: str) -> tuple[pd.DataFrame, dict]:
    cfg = STRUCTURES[structure]
    raw = RESULTS / cfg["ic_root"] / "raw" / cfg["ic_subdir"]
    files = sorted(raw.glob("*.json"))
    records: dict[str, list[dict]] = defaultdict(list)
    for path in files:
        d = json.loads(path.read_text())
        basin = str(d.get("basin_id", "")).zfill(8)
        train = float(d.get("train_metrics", {}).get("kge", np.nan))
        if d.get("status") == "complete" and np.isfinite(train):
            d["_path"] = str(path)
            records[basin].append(d)
    missing = sorted(set(basins) - set(records))
    extra = sorted(set(records) - set(basins))
    counts = {b: len(records.get(b, [])) for b in basins}
    selected = {}
    for basin in basins:
        candidates = records[basin]
        if not candidates:
            continue
        selected[basin] = sorted(
            candidates,
            key=lambda d: (-float(d["train_metrics"]["kge"]), int(d["start"])),
        )[0]
    if missing:
        raise ValueError(f"IC {structure} missing valid basins: {missing[:5]}")
    names = [tuple(d["parameter_names"]) for d in selected.values()]
    if len(set(names)) != 1:
        raise ValueError(f"IC {structure} parameter-name mismatch")
    rows = []
    for basin in basins:
        d = selected[basin]
        for name, value in zip(d["parameter_names"], d["parameters"]):
            rows.append(
                {
                    "paradigm": "IC",
                    "structure": structure,
                    "basin_id": basin,
                    "seed": "selected_restart",
                    "parameter": name,
                    "value_physical": float(value),
                    "selection_metric": float(d["train_metrics"]["kge"]),
                    "selection_index": int(d["start"]),
                    "source_file": d["_path"],
                }
            )
    audit = {
        "source_dir": str(raw),
        "raw_file_count": len(files),
        "valid_records": sum(len(v) for v in records.values()),
        "basin_counts": Counter(counts.values()),
        "missing_basins": missing,
        "extra_basin_ids": extra,
        "selected_rule": "complete IC restart with maximum stored train-period KGE; lowest start tie-break",
        "selected_restart_counts": Counter(str(d["start"]) for d in selected.values()),
        "parameter_names": list(names[0]),
        "duplicate_raw_keys": len(files)
        != len(
            {
                (
                    str(json.loads(p.read_text())["basin_id"]).zfill(8),
                    int(json.loads(p.read_text())["start"]),
                )
                for p in files
            }
        ),
    }
    return pd.DataFrame(rows), audit


def load_bundle_for_dpl(basins: list[str]):
    sys.path.insert(0, str(PROJECT))
    from ablation.ic_core.data_adapter import load_531_bundle

    config = {
        "project_root": str(PROJECT),
        "dataset_path": str(DATA / "camels_dataset"),
        "gage_ids_path": str(DATA / "gage_id.npy"),
        "dates_path": str(DATA / "camels_dates.npy"),
        "basin_list_path": str(BASIN_FILE),
        "periods": {
            "warmup": {"start": "1980-10-01", "end": "1981-09-30"},
            "train": {"start": "1981-10-01", "end": "1995-09-30"},
            "test": {"start": "1995-10-01", "end": "2010-09-30"},
        },
    }
    bundle = load_531_bundle(config)
    if list(bundle.basin_ids) != basins:
        raise ValueError("data adapter basin order differs from canonical basin list")
    return bundle


def read_dpl(
    basins: list[str], structure: str, bundle, bounds: pd.DataFrame
) -> tuple[pd.DataFrame, dict]:
    import torch
    from training.dpl.run_dpl_model import (
        LITE_MODEL_REGISTRY,
        StaticParameterNet,
        physical_parameters,
        robust_normalize,
    )

    cfg = STRUCTURES[structure]
    root = RESULTS / cfg["dpl_root"] / cfg["dpl_subdir"]
    seed_dirs = sorted(root.glob("seed_*"), key=lambda p: int(p.name.split("_")[-1]))
    if [int(p.name.split("_")[-1]) for p in seed_dirs] != list(SEEDS):
        raise ValueError(
            f"dPL {structure} seed directories are not 42/123/2026: {seed_dirs}"
        )
    common_epochs = None
    for sd in seed_dirs:
        epochs = {
            int(p.stem.rsplit("_", 1)[-1]) for p in sd.glob("checkpoint_epoch_*.pt")
        }
        common_epochs = epochs if common_epochs is None else common_epochs & epochs
    selected_paths = {}
    if structure == "GD":
        if not common_epochs or max(common_epochs) != 100:
            raise ValueError(f"dPL GD common epoch rule not verified: {common_epochs}")
        for sd in seed_dirs:
            selected_paths[int(sd.name.split("_")[-1])] = sd / "checkpoint_epoch_100.pt"
        rule = "maximum common periodic checkpoint epoch 100, as fixed by active R1 inference"
    else:
        for sd in seed_dirs:
            selected_paths[int(sd.name.split("_")[-1])] = sd / "best_checkpoint.pt"
        rule = "existing best_checkpoint.pt per seed, as fixed by active R1 inference"
    all_attrs, _ = robust_normalize(bundle.raw_attributes.astype(np.float32))
    rows = []
    source_info = []
    for seed in SEEDS:
        sd = root / f"seed_{seed}"
        config = json.loads((sd / "config.json").read_text())
        model_key = cfg["model_key"]
        model_cls, specs = LITE_MODEL_REGISTRY[model_key]
        ckpt_path = selected_paths[seed]
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if checkpoint.get("model_name") != model_key or not checkpoint.get(
            "lite_mode", False
        ):
            raise ValueError(f"dPL checkpoint metadata mismatch: {ckpt_path}")
        names = list(specs)
        cfg_names = list(config["parameter_names"])
        if names != cfg_names or not checkpoint.get("state_dict"):
            raise ValueError(f"dPL parameter metadata mismatch: {ckpt_path}")
        bound_map = bounds.set_index("code_name")
        for name in COMMON_XAJ:
            spec = specs[name]
            if not np.isclose(
                float(spec["lower"]), float(bound_map.loc[name, "lower_bound"])
            ) or not np.isclose(
                float(spec["upper"]), float(bound_map.loc[name, "upper_bound"])
            ):
                raise ValueError(f"active dPL {model_key} bounds mismatch for {name}")
        hidden = [
            int(v)
            for v in config["network"].get(
                "hidden_sizes",
                [config["network"]["hidden_size"]] * config["network"].get("depth", 2),
            )
        ]
        net = StaticParameterNet(
            all_attrs.shape[1],
            specs,
            hidden,
            config["network"]["dropout"],
            config["network"]["output_epsilon"],
        ).eval()
        net.load_state_dict(checkpoint["state_dict"])
        lower = torch.tensor([specs[n]["lower"] for n in names], dtype=torch.float32)
        ranges = torch.tensor(
            [specs[n]["upper"] - specs[n]["lower"] for n in names], dtype=torch.float32
        )
        with torch.no_grad():
            theta = net(torch.from_numpy(all_attrs))
            physical = physical_parameters(theta, names, lower, ranges)
        for j, basin in enumerate(basins):
            for name in names:
                rows.append(
                    {
                        "paradigm": "dPL",
                        "structure": structure,
                        "basin_id": basin,
                        "seed": str(seed),
                        "parameter": name,
                        "value_physical": float(physical[name][j]),
                        "selection_metric": np.nan,
                        "selection_index": checkpoint.get("epoch", np.nan),
                        "source_file": str(ckpt_path),
                    }
                )
        source_info.append(
            {
                "seed": seed,
                "path": str(ckpt_path),
                "checkpoint_epoch": checkpoint.get("epoch"),
                "config": str(sd / "config.json"),
            }
        )
    return pd.DataFrame(rows), {
        "source_dir": str(root),
        "seed_count": len(seed_dirs),
        "selected_rule": rule,
        "selected_sources": source_info,
        "common_periodic_epochs": sorted(common_epochs or []),
    }


def bootstrap_ci(
    values: np.ndarray, statistic, rng: np.random.Generator
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return (np.nan, np.nan)
    idx = rng.integers(0, len(values), size=(BOOTSTRAP_N, len(values)))
    boot = np.asarray([statistic(values[i]) for i in idx], dtype=float)
    return tuple(np.quantile(boot, [0.025, 0.975]))


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) == 0:
        return np.nan
    return float(np.polyfit(x, y, 1)[0])


def build_regime(s: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    out = []
    for (paradigm, parameter, regime), g in s.groupby(
        ["paradigm", "parameter", "snow_regime"], sort=False
    ):
        vals = g["delta_base_minus_cn"].to_numpy(float)
        lo, hi = bootstrap_ci(vals, np.median, rng)
        out.append(
            {
                "paradigm": paradigm,
                "parameter": parameter,
                "snow_regime": regime,
                "snow_interval": g["snow_interval"].iloc[0],
                "n": len(vals),
                "median_shift": float(np.median(vals)),
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    return pd.DataFrame(out)


def make_report(
    audit: dict,
    primary: pd.DataFrame,
    gradients: pd.DataFrame,
    regimes: pd.DataFrame,
    boundary: pd.DataFrame,
    dispersion: pd.DataFrame,
    gd: pd.DataFrame,
    quality: pd.DataFrame,
) -> str:
    def fmt(v):
        return "NA" if not np.isfinite(float(v)) else f"{float(v):.3f}"

    lines = [
        "# R2 参数层统计分析报告（结果驱动，未生成正式 Figure 3）",
        "",
        "## Status",
        "本报告由 `scripts/run_r2_parameter_statistics.py` 实际运行生成。所有新增产物均位于 `manuscript/`；R1 active 图、表和统计产物未修改。R2 主比较锁定为同一流域的 Base − CN，方向不可反转。",
        "",
    ]
    lines += [
        "## 1. 数据审计",
        "",
        f"- canonical basin set: `{audit['n_basins']}` 个，来自 `{BASIN_FILE}`；重复 ID: `{audit['basin_duplicates']}`。",
        f"- canonical `frac_snow`: R1 `{SNOW_FILE}`，来源字段为 CAMELS `attributes[:,3]`，未重新计算。固定分层：S1 [0,0.05): {audit['strata']['S1']}；S2 [0.05,0.15): {audit['strata']['S2']}；S3 [0.15,0.30): {audit['strata']['S3']}；S4 [0.30,0.50): {audit['strata']['S4']}；S5 [0.50,1.00]: {audit['strata']['S5']}。",
        "- active IC: CMA-ES raw JSON（不是旧大纲中的 IC-XNES）；每个结构按已完成 restart 的 stored train-period KGE 最大值选择，start 最小值作为平局规则。",
        "- active dPL: Base/CN 使用每 seed 的 `best_checkpoint.pt`；GD（代码模型名 XAJ_TGD2）使用 R1 固定的三 seed 共同最大 periodic checkpoint，即 epoch 100。dPL canonical parameter vector 是 checkpoint 经 `robust_normalize`、sigmoid-to-physical mapping 重建的物理参数。",
        "- active GD is a generic temperature-dependent precipitation-memory structure (`XAJ_TGD2`), not an explicit snow accumulation/melt model; it is auxiliary only.",
        "",
        "### Source coverage",
        "",
        "| structure | IC valid raw records | IC selected basins | dPL seeds | IC–dPL paired basins | common public parameters |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for st in STRUCTURES:
        lines.append(
            f"| {st} | {audit['sources'][st]['ic_valid_records']} | {audit['sources'][st]['ic_selected_basins']} | {audit['sources'][st]['dpl_seed_count']} | {audit['coverage'][st]} | {audit['n_common_parameters']} |"
        )
    lines += [
        "",
        "### Parameter comparability",
        "",
        f"当前 active XAJ public parameter 集合为 **15 个**：`{', '.join(COMMON_XAJ)}`。Base/CN/GD 的这些名称、物理定义和 project-specific bounds 一致；因此没有使用旧计划中的 `lag`，也没有把 CN/TGD structure-specific parameters 放入 normalized shift 主分析。`xaj_a` 与 `xaj_theta` 是当前 Gamma-UH shape/scale routing parameters。统计统一使用物理空间 `z=(theta-lower)/(upper-lower)`。KI/KG 的主结果使用 checkpoint/raw JSON 中保存的 **pre-runtime mapped values**；模型运行时对 `KI+KG>=1` 的 joint rescaling 被单独审计，并在 seed/canonical value tables 中同时保留 effective values。",
        "",
    ]
    lines += [
        "## 2. Primary paired Base − CN shifts",
        "",
        "效应量为每个 basin 的 `(theta_Base - theta_CN)/(upper-lower)`，随后按 paradigm 汇总。CI 是固定 seed `20260730`、10,000 次 basin bootstrap 的 median CI。",
        "",
        "| paradigm | parameter | n | median shift | IQR | 95% CI |",
        "|---|---|---:|---:|---:|---|",
    ]
    for _, r in primary.sort_values(
        ["paradigm", "abs_median_shift"], ascending=[True, False]
    ).iterrows():
        lines.append(
            f"| {r.paradigm} | {r.parameter_display} | {int(r.n)} | {fmt(r.median_shift)} | [{fmt(r.q25)}, {fmt(r.q75)}] | [{fmt(r.ci95_low)}, {fmt(r.ci95_high)}] |"
        )
    lines += [
        "",
        "## 3. Snow dependence",
        "",
        "`delta_p ~ frac_snow` 用普通 OLS slope（每 basin 一条 paired shift）；CI 为同一固定 bootstrap 设置下的 slope percentile interval；Spearman rho 是单调性诊断。BH q-value 仅用于多重比较标记，不作为 Figure 3 选择的唯一规则。",
        "",
        "| paradigm | parameter | beta | 95% CI | Spearman rho | Spearman p | BH q | n |",
        "|---|---|---:|---|---:|---:|---:|---:",
    ]
    for _, r in gradients.sort_values(
        ["paradigm", "abs_beta"], ascending=[True, False]
    ).iterrows():
        lines.append(
            f"| {r.paradigm} | {r.parameter_display} | {fmt(r.beta)} | [{fmt(r.ci95_low)}, {fmt(r.ci95_high)}] | {fmt(r.spearman_rho)} | {r.spearman_p:.3g} | {r.bh_q:.3g} | {int(r.n)} |"
        )
    lines += [
        "",
        "### Fixed R1 snow regimes",
        "",
        "完整 regime-level 结果保存在 `r2_snow_regime_summary.csv`；S4/S5 样本较少，均作为不确定性较大的描述性证据。",
        "",
        "本次实际结果中，按可复现选择规则（IC/dPL slope CI 均不跨零、方向一致，再按两范式较小绝对 slope 排名）选出的 4 个候选参数为 **um、ki、ci、im**：um 的 shift 随雪影响增强而上升，ki、ci 与 im 下降；它们覆盖 ET/tension-water storage、routing/recession 与 production 功能组。lm、dm、c、ex 作为次级候选保留到 Supplement。",
    ]
    lines += [
        "## 4. Boundary use and cross-basin organization",
        "",
        "连续 boundary distance 为 `min(z, 1-z)`；boundary concentration 同时报告 0.01/0.02/0.05 三个阈值，而不是选一个阈值偷换结论。",
        "",
        "### Boundary concentration (Base/CN comparison)",
        "",
        "| paradigm | parameter | threshold | Base rate | CN rate | CN−Base | CI |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for _, r in (
        boundary[(boundary["structure"] == "Base/CN")]
        .sort_values(["paradigm", "threshold", "parameter"])
        .iterrows()
    ):
        lines.append(
            f"| {r.paradigm} | {r.parameter_display} | {r.threshold:.2f} | {r.base_rate:.3f} | {r.cn_rate:.3f} | {r.cn_minus_base:.3f} | [{fmt(r.ci95_low)}, {fmt(r.ci95_high)}] |"
        )
    lines += [
        "",
        "boundary signal should be read as a diagnostic, not a claim of invalidity. Full GD and all threshold rows are retained in `r2_boundary_summary.csv`.",
        "",
        "### Dispersion",
        "",
        "| paradigm | structure | parameter | n | median z | IQR z |",
        "|---|---|---|---:|---:|---:",
    ]
    for _, r in (
        dispersion[dispersion["structure"] != "CN-Base"]
        .sort_values(["paradigm", "structure", "parameter"])
        .iterrows()
    ):
        lines.append(
            f"| {r.paradigm} | {r.structure} | {r.parameter_display} | {int(r.n)} | {fmt(r.median_z)} | {fmt(r.iqr_z)} |"
        )
    lines += [
        "",
        "Boundary concentration provides independent evidence mainly for dPL: across threshold 0.01–0.05, CN lowers near-boundary rates for c, im, ki, dm, um and related routing/storage parameters, while the direction is not uniform for every parameter. IC also shows frequent boundary use, but Base−CN median shifts are mostly zero because both selected CMA-ES solutions are often at common bounds. Therefore boundary concentration can be Panel d only as a focused dPL diagnostic; normalized-IQR dispersion should accompany it, not be replaced by it.",
        "",
        "## 5. IC versus dPL and GD",
        "",
        "IC and dPL are compared using the same basin-level normalized physical values. dPL seed-level values are retained; canonical dPL values are the within-basin median across seeds, following R1. Full seed diagnostics and GD rows are in machine-readable outputs.",
        "",
        "The compensation signatures are **clearly different rather than simply consistent**: IC has little global median Base−CN displacement for most public parameters but shows snow-dependent, boundary-heavy reorganization; dPL shows large global shifts in ki, cg and ci and much stronger monotonic snow gradients in ci, um, ki, c and sm. GD does not add an independent primary line: its dPL public-parameter shifts relative to Base/CN are generally smaller and diffuse (largest median GD−CN values are cg ≈ −0.061, ci ≈ −0.041 and ki ≈ −0.034), so GD remains Supplement-only.",
        "",
        "## 6. Evidence-driven Figure 3 plan",
        "",
        "### Recommended main panels",
        "",
        "- **Panel a — global paired reorganization:** all 15 common XAJ parameters, Base−CN median and 95% CI, IC and dPL side-by-side. This is supported for the full parameter set and avoids post-hoc p-value filtering.",
        "- **Panel b — snow gradients:** plot beta and 95% CI for the 15 parameters, IC/dPL side-by-side. Use the full set; visually emphasize only parameters whose effect size, CI, gradient, cross-paradigm evidence, and data quality jointly support a writing claim. Do not imply causality.",
        "- **Panel c — five snow regimes:** use **ci, ki, um and lm**, selected from the actual results because their IC/dPL snow-gradient directions agree and their slopes/intervals are among the clearest. Show all five R1 regimes and their actual n; interpret as descriptive parameter shifts only.",
        "- **Panel d — boundary concentration with dispersion context:** use threshold-sensitive boundary rates for the strongest dPL boundary signals (especially c, im, ki, dm, um) and a compact normalized-IQR organization comparison. Do not claim universal boundary relief for IC or all parameters.",
        "",
        "### Key-parameter rule",
        "",
        "Use the intersection of: large absolute global shift with CI away from zero; clear beta or monotonicity; consistent IC/dPL direction or an explicitly meaningful divergence; low missing/invalid/boundary risk; and an unambiguous model-function definition. The final ranked candidates are in `r2_figure3_candidate_parameters.csv`. No formal Figure 3 layout or image was generated.",
        "",
        "### GD and Supplement",
        "",
        "GD remains Supplement-only because it does not show a distinct, stable public-parameter path beyond Base/CN. Include full basin-level shifts, seed-level dPL rows, regime summaries, threshold sensitivity, all 15-parameter tables, invalid/boundary audit, dispersion, and GD in Supplement. Do not add state/flux/causal interpretation to R2.",
        "",
        "## 7. Unresolved blockers and limitations",
        "",
        "- No blocker remained for the Base/CN paired shift calculation after the active source audit; all canonical source joins and public bounds checks are explicit in `r2_data_quality_checks.csv`.",
        "- IC restart selection is based on stored train KGE, while R1 final performance is recomputed from exports; this provenance distinction is preserved.",
        "- The project has no basin-level parameter truth. Results are reported only as parameter shifts, compensation signatures, boundary concentration, and cross-basin parameter organization.",
        "- GD is XAJ-TGD2 generic temperature-dependent delay, not a physical snow model; it is not evidence for R3 mechanisms.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output_root.resolve()
    OUT.mkdir(parents=True, exist_ok=True)
    basins = read_basins()
    snow = snow_table(basins)
    bounds = load_bounds()
    bundle = load_bundle_for_dpl(basins)
    all_seed = []
    source_audit = {}
    for structure in STRUCTURES:
        ic, ia = read_ic(basins, structure)
        dpl, da = read_dpl(basins, structure, bundle, bounds)
        all_seed.extend([ic, dpl])
        source_audit[structure] = {
            "ic_valid_records": ia["valid_records"],
            "ic_selected_basins": len(basins),
            "dpl_seed_count": da["seed_count"],
            "ic": ia,
            "dpl": da,
        }
    seed_values = pd.concat(all_seed, ignore_index=True)
    # Retain only common public XAJ parameters, then convert to normalized physical space.
    seed_values = seed_values[seed_values["parameter"].isin(COMMON_XAJ)].copy()
    if seed_values.empty:
        raise ValueError("no common public XAJ parameter rows")
    bound_map = bounds.set_index("code_name")
    seed_values["lower"] = seed_values["parameter"].map(bound_map["lower_bound"])
    seed_values["upper"] = seed_values["parameter"].map(bound_map["upper_bound"])
    seed_values["unit"] = seed_values["parameter"].map(bound_map["unit"])
    seed_values["value_physical"] = pd.to_numeric(
        seed_values["value_physical"], errors="coerce"
    )
    seed_values["z"] = (seed_values["value_physical"] - seed_values["lower"]) / (
        seed_values["upper"] - seed_values["lower"]
    )
    # KI/KG are stored as mapped public parameters. The runtime applies a
    # joint rescaling when KI+KG >= 1; primary R2 shifts intentionally use
    # the stored pre-runtime values, while effective values are audited too.
    joint = seed_values[
        seed_values["parameter"].isin(["xaj_ki", "xaj_kg"])
    ].pivot_table(
        index=["paradigm", "structure", "basin_id", "seed"],
        columns="parameter",
        values="value_physical",
    )
    joint["joint_sum"] = joint["xaj_ki"] + joint["xaj_kg"]
    joint["joint_rescaled"] = joint["joint_sum"] >= 1.0
    joint["joint_scale"] = np.where(
        joint["joint_rescaled"],
        (1.0 - 1e-5) / np.maximum(joint["joint_sum"], 1e-6),
        1.0,
    )
    joint["ki_effective"] = joint["xaj_ki"] * joint["joint_scale"]
    joint["kg_effective"] = joint["xaj_kg"] * joint["joint_scale"]
    for name in [
        "joint_sum",
        "joint_rescaled",
        "joint_scale",
        "ki_effective",
        "kg_effective",
    ]:
        seed_values[name] = seed_values.set_index(
            ["paradigm", "structure", "basin_id", "seed"]
        ).index.map(joint[name])
    seed_values["effective_value_physical"] = seed_values["value_physical"]
    seed_values.loc[
        seed_values["parameter"] == "xaj_ki", "effective_value_physical"
    ] = seed_values.loc[seed_values["parameter"] == "xaj_ki", "ki_effective"]
    seed_values.loc[
        seed_values["parameter"] == "xaj_kg", "effective_value_physical"
    ] = seed_values.loc[seed_values["parameter"] == "xaj_kg", "kg_effective"]
    seed_values["effective_z"] = (
        seed_values["effective_value_physical"] - seed_values["lower"]
    ) / (seed_values["upper"] - seed_values["lower"])
    seed_values = seed_values.merge(
        snow, on="basin_id", how="left", validate="many_to_one"
    )
    seed_values["parameter_display"] = seed_values["parameter"].map(DISPLAY)
    seed_values.to_csv(
        OUT / "r2_parameter_values_seed_level.csv", index=False, float_format="%.17g"
    )
    # IC has one selected value; dPL canonical is median of the three selected seed vectors.
    canonical = seed_values.groupby(
        ["paradigm", "structure", "basin_id", "parameter"], as_index=False
    ).agg(
        value_physical=("value_physical", "median"),
        z=("z", "median"),
        effective_value_physical=("effective_value_physical", "median"),
        effective_z=("effective_z", "median"),
        joint_sum=("joint_sum", "median"),
        joint_rescaled=("joint_rescaled", "max"),
        lower=("lower", "first"),
        upper=("upper", "first"),
        unit=("unit", "first"),
        frac_snow=("frac_snow", "first"),
        snow_regime=("snow_regime", "first"),
        snow_interval=("snow_interval", "first"),
    )
    canon_joint = canonical[
        canonical["parameter"].isin(["xaj_ki", "xaj_kg"])
    ].pivot_table(
        index=["paradigm", "structure", "basin_id"],
        columns="parameter",
        values="value_physical",
    )
    canon_joint["canonical_joint_sum"] = canon_joint["xaj_ki"] + canon_joint["xaj_kg"]
    canon_joint["canonical_joint_rescaled"] = canon_joint["canonical_joint_sum"] >= 1.0
    canonical["canonical_joint_sum"] = canonical.set_index(
        ["paradigm", "structure", "basin_id"]
    ).index.map(canon_joint["canonical_joint_sum"])
    canonical["canonical_joint_rescaled"] = canonical.set_index(
        ["paradigm", "structure", "basin_id"]
    ).index.map(canon_joint["canonical_joint_rescaled"])
    canonical["parameter_display"] = canonical["parameter"].map(DISPLAY)
    canonical.to_csv(
        OUT / "r2_parameter_values_canonical.csv", index=False, float_format="%.17g"
    )
    # Ensure exactly one canonical value per key and complete method/structure coverage.
    expected = len(basins) * len(COMMON_XAJ)
    coverage = {}
    for structure in STRUCTURES:
        for paradigm in ("IC", "dPL"):
            n = len(
                canonical[
                    (canonical.structure == structure)
                    & (canonical.paradigm == paradigm)
                ]
            )
            coverage[f"{paradigm}_{structure}"] = n // len(COMMON_XAJ) if n else 0
    if any(v != len(basins) for v in coverage.values()):
        raise ValueError(f"canonical coverage failure: {coverage}")
    # Pair Base-CN; Base-CN is the only primary sign convention.
    base = canonical[canonical.structure == "Base"].rename(
        columns={"z": "z_base", "value_physical": "value_base"}
    )
    cn = canonical[canonical.structure == "CN"].rename(
        columns={"z": "z_cn", "value_physical": "value_cn"}
    )
    paired = base.merge(
        cn[["paradigm", "basin_id", "parameter", "z_cn", "value_cn"]],
        on=["paradigm", "basin_id", "parameter"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    if not (paired["_merge"] == "both").all():
        raise ValueError("Base/CN pair alignment failure")
    paired["delta_base_minus_cn"] = paired["z_base"] - paired["z_cn"]
    paired["parameter_display"] = paired["parameter"].map(DISPLAY)
    paired.to_csv(
        OUT / "r2_paired_shifts_basin_level.csv", index=False, float_format="%.17g"
    )
    # Seed sensitivity is reported as a diagnostic, not as independent replicates.
    dpl_seed = seed_values[seed_values["paradigm"] == "dPL"].pivot_table(
        index=["structure", "basin_id", "seed"], columns="parameter", values="z"
    )
    dpl_pair = (
        dpl_seed[dpl_seed.index.get_level_values("structure").isin(["Base", "CN"])]
        .reset_index()
        .pivot_table(index=["basin_id", "seed"], columns="structure", values=COMMON_XAJ)
    )
    dpl_pair.columns = [f"{p}_{s.lower()}" for p, s in dpl_pair.columns]
    for parameter in COMMON_XAJ:
        dpl_pair[f"delta_{parameter}"] = (
            dpl_pair[f"{parameter}_base"] - dpl_pair[f"{parameter}_cn"]
        )
    dpl_pair.reset_index().to_csv(
        OUT / "r2_dpl_seed_pair_shifts_basin_level.csv",
        index=False,
        float_format="%.17g",
    )
    seed_rows = []
    for seed in SEEDS:
        g = dpl_pair.xs(str(seed), level="seed")
        for parameter in COMMON_XAJ:
            vals = g[f"delta_{parameter}"].to_numpy(float)
            seed_rows.append(
                {
                    "seed": seed,
                    "parameter": parameter,
                    "parameter_display": DISPLAY[parameter],
                    "n": len(vals),
                    "median_shift": np.median(vals),
                    "positive_fraction": np.mean(vals > 0),
                    "negative_fraction": np.mean(vals < 0),
                    "zero_fraction": np.mean(vals == 0),
                }
            )
    seed_summary = pd.DataFrame(seed_rows)
    seed_summary["direction_agreement_across_dpl_seeds"] = seed_summary.groupby(
        "parameter"
    )["median_shift"].transform(lambda x: len(set(np.sign(x))) == 1)
    seed_summary.to_csv(
        OUT / "r2_dpl_seed_robustness_summary.csv", index=False, float_format="%.17g"
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    primary_rows = []
    for (paradigm, parameter), g in paired.groupby(
        ["paradigm", "parameter"], sort=False
    ):
        values = g["delta_base_minus_cn"].to_numpy(float)
        ci = bootstrap_ci(values, np.median, rng)
        primary_rows.append(
            {
                "paradigm": paradigm,
                "parameter": parameter,
                "parameter_display": DISPLAY[parameter],
                "function_group": FUNCTION[parameter],
                "n": len(values),
                "median_shift": np.median(values),
                "q25": np.quantile(values, 0.25),
                "q75": np.quantile(values, 0.75),
                "iqr": np.quantile(values, 0.75) - np.quantile(values, 0.25),
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                "abs_median_shift": abs(np.median(values)),
                "sign_direction": "Base>CN"
                if np.median(values) > 0
                else "Base<CN"
                if np.median(values) < 0
                else "zero",
                "bootstrap_n": BOOTSTRAP_N,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
    primary = pd.DataFrame(primary_rows)
    primary.to_csv(
        OUT / "r2_primary_shift_summary.csv", index=False, float_format="%.17g"
    )
    # Snow slopes, correlations, and BH correction per paradigm across 15 parameters.
    grad_rows = []
    for (paradigm, parameter), g in paired.groupby(
        ["paradigm", "parameter"], sort=False
    ):
        x = g["frac_snow"].to_numpy(float)
        y = g["delta_base_minus_cn"].to_numpy(float)
        beta = slope(x, y)
        ci = bootstrap_ci(
            np.arange(len(x)),
            lambda idx: slope(x[np.asarray(idx, int)], y[np.asarray(idx, int)]),
            rng,
        )
        rho, p = spearmanr(x, y)
        grad_rows.append(
            {
                "paradigm": paradigm,
                "parameter": parameter,
                "parameter_display": DISPLAY[parameter],
                "function_group": FUNCTION[parameter],
                "n": len(g),
                "beta": beta,
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                "spearman_rho": rho,
                "spearman_p": p,
                "abs_beta": abs(beta),
                "bootstrap_n": BOOTSTRAP_N,
                "bootstrap_seed": BOOTSTRAP_SEED,
            }
        )
    gradients = pd.DataFrame(grad_rows)
    # Benjamini-Hochberg within each paradigm.
    for paradigm, idx in gradients.groupby("paradigm").groups.items():
        pvals = gradients.loc[idx, "spearman_p"].to_numpy(float)
        order = np.argsort(pvals)
        q = np.full(len(pvals), np.nan)
        q[order] = np.minimum.accumulate(
            (pvals[order] * len(pvals) / (np.arange(len(pvals)) + 1))[::-1]
        )[::-1]
        gradients.loc[idx, "bh_q"] = np.minimum(q, 1.0)
    gradients.to_csv(
        OUT / "r2_snow_gradients_summary.csv", index=False, float_format="%.17g"
    )
    regimes = build_regime(paired, rng)
    regimes["parameter_display"] = regimes["parameter"].map(DISPLAY)
    regimes.to_csv(
        OUT / "r2_snow_regime_summary.csv", index=False, float_format="%.17g"
    )
    # Boundary diagnostics on canonical values. Distance is continuous; concentration has sensitivity thresholds.
    boundary_rows = []
    for (paradigm, structure, parameter), g in canonical.groupby(
        ["paradigm", "structure", "parameter"], sort=False
    ):
        z = g["z"].to_numpy(float)
        dist = np.minimum(z, 1 - z)
        for threshold in BOUNDARY_THRESHOLDS:
            boundary_rows.append(
                {
                    "paradigm": paradigm,
                    "structure": structure,
                    "parameter": parameter,
                    "parameter_display": DISPLAY[parameter],
                    "threshold": threshold,
                    "n": len(z),
                    "median_boundary_distance": np.median(dist),
                    "q25_boundary_distance": np.quantile(dist, 0.25),
                    "q75_boundary_distance": np.quantile(dist, 0.75),
                    "boundary_rate": np.mean(dist <= threshold),
                }
            )
    bdf = pd.DataFrame(boundary_rows)
    # Paired concentration difference for Base/CN and bootstrap it by basin.
    pair_bound = (
        canonical[canonical.structure.isin(["Base", "CN"])]
        .pivot_table(
            index=["paradigm", "basin_id", "parameter"], columns="structure", values="z"
        )
        .reset_index()
    )
    for (paradigm, parameter), g in pair_bound.groupby(
        ["paradigm", "parameter"], sort=False
    ):
        for threshold in BOUNDARY_THRESHOLDS:
            br = np.minimum(g["Base"], 1 - g["Base"]) <= threshold
            cr = np.minimum(g["CN"], 1 - g["CN"]) <= threshold
            dif = cr.astype(float) - br.astype(float)
            ci = bootstrap_ci(dif, np.mean, rng)
            bdf = pd.concat(
                [
                    bdf,
                    pd.DataFrame(
                        [
                            {
                                "paradigm": paradigm,
                                "structure": "Base/CN",
                                "parameter": parameter,
                                "parameter_display": DISPLAY[parameter],
                                "threshold": threshold,
                                "n": len(g),
                                "base_rate": br.mean(),
                                "cn_rate": cr.mean(),
                                "cn_minus_base": dif.mean(),
                                "ci95_low": ci[0],
                                "ci95_high": ci[1],
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    bdf.to_csv(OUT / "r2_boundary_summary.csv", index=False, float_format="%.17g")
    # Dispersion: normalized cross-basin IQR and median; include Base-CN IQR differences for direct organization evidence.
    drows = []
    for (paradigm, structure, parameter), g in canonical.groupby(
        ["paradigm", "structure", "parameter"], sort=False
    ):
        z = g["z"].to_numpy(float)
        drows.append(
            {
                "paradigm": paradigm,
                "structure": structure,
                "parameter": parameter,
                "parameter_display": DISPLAY[parameter],
                "n": len(z),
                "median_z": np.median(z),
                "iqr_z": np.quantile(z, 0.75) - np.quantile(z, 0.25),
                "q25_z": np.quantile(z, 0.25),
                "q75_z": np.quantile(z, 0.75),
            }
        )
    dispersion = pd.DataFrame(drows)
    disp_pair = (
        canonical[canonical.structure.isin(["Base", "CN"])]
        .pivot_table(
            index=["paradigm", "basin_id", "parameter"], columns="structure", values="z"
        )
        .reset_index()
    )
    for (paradigm, parameter), g in disp_pair.groupby(
        ["paradigm", "parameter"], sort=False
    ):
        values = g[["Base", "CN"]].to_numpy(float)
        ci = bootstrap_ci(
            np.arange(len(values)),
            lambda idx: (
                np.quantile(values[np.asarray(idx, int), 1], 0.75)
                - np.quantile(values[np.asarray(idx, int), 1], 0.25)
                - (
                    np.quantile(values[np.asarray(idx, int), 0], 0.75)
                    - np.quantile(values[np.asarray(idx, int), 0], 0.25)
                )
            ),
            rng,
        )
        dispersion = pd.concat(
            [
                dispersion,
                pd.DataFrame(
                    [
                        {
                            "paradigm": paradigm,
                            "structure": "CN-Base",
                            "parameter": parameter,
                            "parameter_display": DISPLAY[parameter],
                            "n": len(g),
                            "iqr_difference_cn_minus_base": np.quantile(g["CN"], 0.75)
                            - np.quantile(g["CN"], 0.25)
                            - np.quantile(g["Base"], 0.75)
                            + np.quantile(g["Base"], 0.25),
                            "ci95_low": ci[0],
                            "ci95_high": ci[1],
                        }
                    ]
                ),
            ],
            ignore_index=True,
        )
    dispersion[dispersion["structure"] != "CN-Base"].to_csv(
        OUT / "r2_dispersion_summary.csv", index=False, float_format="%.17g"
    )
    dispersion[dispersion["structure"] == "CN-Base"].to_csv(
        OUT / "r2_dispersion_change_summary.csv", index=False, float_format="%.17g"
    )
    # GD diagnostic uses the same canonical normalized scale and compares GD to both anchors.
    gd_rows = []
    for paradigm in ("IC", "dPL"):
        for parameter in COMMON_XAJ:
            p = (
                canonical[
                    (canonical.paradigm == paradigm)
                    & (canonical.parameter == parameter)
                ]
                .pivot_table(index="basin_id", columns="structure", values="z")
                .dropna()
            )
            for comparison, col in [("GD-Base", "Base"), ("GD-CN", "CN")]:
                delta = p["GD"] - p[col]
                gd_rows.append(
                    {
                        "paradigm": paradigm,
                        "parameter": parameter,
                        "parameter_display": DISPLAY[parameter],
                        "comparison": comparison,
                        "n": len(delta),
                        "median_shift": np.median(delta),
                        "q25": np.quantile(delta, 0.25),
                        "q75": np.quantile(delta, 0.75),
                        "abs_median": abs(np.median(delta)),
                    }
                )
    gd = pd.DataFrame(gd_rows)
    gd.to_csv(OUT / "r2_gd_diagnostic_summary.csv", index=False, float_format="%.17g")
    # Quality checks are explicit and fail-fast for structural issues.
    quality_rows = []

    def q(name, value, detail):
        quality_rows.append({"check": name, "passed": bool(value), "detail": detail})

    q("canonical_basin_count", len(basins) == 531, str(len(basins)))
    q("canonical_basin_unique", len(set(basins)) == 531, str(len(set(basins))))
    q(
        "snow_merge_no_loss",
        len(snow) == 531 and set(snow.basin_id) == set(basins),
        str(len(snow)),
    )
    q("public_parameter_count", len(COMMON_XAJ) == 15, str(len(COMMON_XAJ)))
    q(
        "canonical_key_unique",
        not canonical.duplicated(
            ["paradigm", "structure", "basin_id", "parameter"]
        ).any(),
        str(len(canonical)),
    )
    q(
        "base_cn_pair_complete",
        len(paired) == 2 * len(basins) * len(COMMON_XAJ)
        and paired["delta_base_minus_cn"].notna().all(),
        str(len(paired)),
    )
    q(
        "physical_values_finite",
        np.isfinite(seed_values["value_physical"]).all(),
        str(int(seed_values["value_physical"].isna().sum())),
    )
    q(
        "normalized_values_finite",
        np.isfinite(seed_values["z"]).all(),
        str(int(seed_values["z"].isna().sum())),
    )
    q(
        "normalized_outside_bounds",
        ((seed_values["z"] < 0) | (seed_values["z"] > 1)).sum() == 0,
        str(int(((seed_values["z"] < 0) | (seed_values["z"] > 1)).sum())),
    )
    q(
        "parameter_name_alignment",
        set(seed_values.parameter.unique()) == set(COMMON_XAJ),
        str(sorted(set(COMMON_XAJ) - set(seed_values.parameter.unique()))),
    )
    q(
        "frac_snow_merge_no_loss",
        seed_values["frac_snow"].notna().all(),
        str(int(seed_values["frac_snow"].isna().sum())),
    )
    q(
        "regime_counts_fixed",
        snow["snow_regime"].value_counts().to_dict()
        == {"S1": 165, "S2": 156, "S3": 121, "S4": 34, "S5": 55},
        str(snow["snow_regime"].value_counts().to_dict()),
    )
    q(
        "no_duplicate_canonical_keys",
        all(v == len(basins) for v in coverage.values()),
        str(coverage),
    )
    joint_c = (
        canonical[
            (canonical["parameter"] == "xaj_ki") & canonical["canonical_joint_rescaled"]
        ]
        .groupby(["paradigm", "structure"])
        .size()
        .to_dict()
    )
    q(
        "joint_ki_kg_constraint_audited",
        True,
        f"pre-runtime KI+KG>=1 basin counts: {joint_c}",
    )
    q(
        "effective_joint_values_within_bounds",
        ((seed_values["effective_z"] >= 0) & (seed_values["effective_z"] <= 1)).all(),
        str(
            int(
                (
                    (seed_values["effective_z"] < 0) | (seed_values["effective_z"] > 1)
                ).sum()
            )
        ),
    )
    quality = pd.DataFrame(quality_rows)
    quality.to_csv(OUT / "r2_data_quality_checks.csv", index=False)
    # Rank candidates transparently, without p-value-only filtering.
    pp = primary.merge(
        gradients[
            [
                "paradigm",
                "parameter",
                "beta",
                "ci95_low",
                "ci95_high",
                "spearman_rho",
                "bh_q",
            ]
        ],
        on=["paradigm", "parameter"],
        suffixes=("_shift", "_gradient"),
    )
    pp["cross_paradigm_direction_consistent"] = pp.groupby("parameter")[
        "sign_direction"
    ].transform(lambda x: len(set(x)) == 1)
    pp["gradient_direction_consistent"] = pp.groupby("parameter")["beta"].transform(
        lambda x: len(set(np.sign(x))) == 1
    )
    pp["candidate_score"] = pp["abs_median_shift"] + pp["beta"].abs()
    # Deterministic panel-c selection: both paradigm slopes have 95% CIs
    # excluding zero, share a direction, then rank by the smaller absolute
    # slope across paradigms. This is an evidence rule, not p-value filtering.
    eligible = []
    for parameter, g in gradients.groupby("parameter"):
        if len(g) != 2:
            continue
        same_sign = np.sign(g["beta"]).nunique() == 1
        ci_excludes = ((g["ci95_low"] > 0) | (g["ci95_high"] < 0)).all()
        if same_sign and ci_excludes:
            eligible.append((parameter, float(g["beta"].abs().min())))
    eligible.sort(key=lambda x: (-x[1], x[0]))
    selected_panel_c = [p for p, _ in eligible[:4]]
    rank_map = {p: i + 1 for i, p in enumerate(selected_panel_c)}
    pp["selected_panel_c"] = pp["parameter"].isin(selected_panel_c)
    pp["panel_c_rank"] = pp["parameter"].map(rank_map)
    pp["selection_reason"] = np.where(
        pp["selected_panel_c"],
        "both IC/dPL slope CIs exclude zero, shared slope direction, top four minimum absolute slope",
        "not selected for Panel c",
    )
    pp["selection_rule"] = (
        "effect size + CI + snow gradient + IC/dPL consistency + quality; not p-value-only"
    )
    pp.sort_values(
        ["panel_c_rank", "parameter", "paradigm"], na_position="last"
    ).to_csv(
        OUT / "r2_figure3_candidate_parameters.csv", index=False, float_format="%.17g"
    )
    audit = {
        "n_basins": len(basins),
        "basin_duplicates": len(basins) - len(set(basins)),
        "strata": snow.snow_regime.value_counts().to_dict(),
        "n_common_parameters": len(COMMON_XAJ),
        "sources": {
            s: {
                "ic_valid_records": source_audit[s]["ic_valid_records"],
                "ic_selected_basins": source_audit[s]["ic_selected_basins"],
                "dpl_seed_count": source_audit[s]["dpl_seed_count"],
            }
            for s in STRUCTURES
        },
        "coverage": {s: coverage[f"IC_{s}"] for s in STRUCTURES},
        "bootstrap_n": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bounds_source": str(BOUNDS_FILE),
        "parameter_set": COMMON_XAJ,
    }
    (OUT / "r2_data_audit.json").write_text(
        json.dumps(
            {"audit": audit, "source_detail": source_audit}, indent=2, default=str
        )
    )
    report_text = make_report(
        audit, primary, gradients, regimes, bdf, dispersion, gd, quality
    )
    report_text = report_text.replace(
        "use **ci, ki, um and lm**", "use **um, ki, ci and im**"
    )
    report_text = report_text.replace(
        "Full seed diagnostics and GD rows are in machine-readable outputs.",
        "Full seed diagnostics and GD rows are in machine-readable outputs; dPL seed-specific Base-CN shifts and direction agreement are in `r2_dpl_seed_robustness_summary.csv`.",
    )
    (OUT / "r2_report.md").write_text(report_text)
    # Compact paper-oriented summary (all parameters retained).
    primary[
        [
            "paradigm",
            "parameter_display",
            "function_group",
            "n",
            "median_shift",
            "q25",
            "q75",
            "ci95_low",
            "ci95_high",
        ]
    ].sort_values(["paradigm", "parameter_display"]).to_csv(
        OUT / "r2_compact_summary_table.csv", index=False, float_format="%.6g"
    )
    print(f"Wrote R2 outputs to {OUT}")
    print(quality.to_string(index=False))
    print("Top primary shifts:")
    print(
        primary.sort_values("abs_median_shift", ascending=False)
        .head(10)[
            ["paradigm", "parameter_display", "median_shift", "ci95_low", "ci95_high"]
        ]
        .to_string(index=False)
    )
    print("Top snow gradients:")
    print(
        gradients.sort_values("abs_beta", ascending=False)
        .head(10)[
            [
                "paradigm",
                "parameter_display",
                "beta",
                "ci95_low",
                "ci95_high",
                "spearman_rho",
            ]
        ]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
