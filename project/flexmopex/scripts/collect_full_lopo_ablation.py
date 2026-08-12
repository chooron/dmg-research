from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
DATA_ROOT = REPO_ROOT / "data"
os.environ.setdefault("FLEXMOPEX_DATA_DIR", str(DATA_ROOT))
os.environ.setdefault("DATA_PATH", str(DATA_ROOT))
os.environ.setdefault("BASIN_GROUPS_DIR", str(DATA_ROOT / "basin_groups"))
os.environ.setdefault("GAGE_INFO", str(DATA_ROOT / "gage_id.npy"))
DEFAULT_ROOT = PROJECT_DIR / "results" / "block1_full_lopo"
DEFAULT_FULL_ROOT = PROJECT_DIR / "results" / "block1_main" / "full" / "alpha0.0"
GAGE_ID_PATH = REPO_ROOT / "data" / "gage_id.npy"

PROCESS_SPECS = {
    "phenology": "full_minus_phenology",
    "interception": "full_minus_interception",
    "snow": "full_minus_snow",
    "subsurface": "full_minus_subsurface",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect Full-MOPEX LOPO ablation results.")
    parser.add_argument("--root", default=str(DEFAULT_ROOT), help="LOPO results root.")
    parser.add_argument(
        "--full-root",
        default=str(DEFAULT_FULL_ROOT),
        help="Existing Full-MOPEX reference root.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--test-epoch", type=int, default=50)
    return parser.parse_args()


def read_metrics_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, str):
        data = json.loads(data)
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not decode to a metrics dictionary.")
    return data


def load_gage_ids() -> np.ndarray:
    if not GAGE_ID_PATH.exists():
        raise FileNotFoundError(f"gage_id file not found: {GAGE_ID_PATH}")
    return np.load(GAGE_ID_PATH).astype(str)


def find_test_dir(base_dir: Path) -> Path:
    candidates = sorted(base_dir.rglob("test*_Ep*/metrics.json"))
    if not candidates:
        raise FileNotFoundError(f"No test metrics.json found under {base_dir}")
    return candidates[-1].parent


def full_seed_base(full_root: Path, seed: int) -> Path:
    base = full_root / f"seed{seed}"
    if not base.exists():
        raise FileNotFoundError(f"Full reference seed directory not found: {base}")
    return base


def ablation_seed_base(root: Path, ablation_name: str, seed: int) -> Path:
    base = root / "config_dmopex_v1" / ablation_name / f"seed_{seed}"
    if not base.exists():
        raise FileNotFoundError(f"Ablation seed directory not found: {base}")
    return base


def load_seed_nse(metrics_path: Path, gage_ids: np.ndarray) -> pd.DataFrame:
    metrics = read_metrics_json(metrics_path)
    nse = np.asarray(metrics.get("nse", []), dtype=float).ravel()
    if nse.size != gage_ids.size:
        raise ValueError(
            f"{metrics_path} has {nse.size} NSE values but {gage_ids.size} gage IDs."
        )
    return pd.DataFrame(
        {
            "basin_idx": np.arange(gage_ids.size, dtype=int),
            "gage_id": gage_ids,
            "nse": nse,
        }
    )


def collect_seed_level(root: Path, full_root: Path, seeds: list[int]) -> tuple[pd.DataFrame, dict[str, Path]]:
    gage_ids = load_gage_ids()
    rows: list[pd.DataFrame] = []
    source_paths: dict[str, Path] = {}
    for process, ablation_name in PROCESS_SPECS.items():
        for seed in seeds:
            full_test_dir = find_test_dir(full_seed_base(full_root, seed))
            ablation_test_dir = find_test_dir(ablation_seed_base(root, ablation_name, seed))
            source_paths[f"full_seed_{seed}"] = full_test_dir
            source_paths[f"{ablation_name}_seed_{seed}"] = ablation_test_dir

            full_df = load_seed_nse(full_test_dir / "metrics.json", gage_ids).rename(
                columns={"nse": "nse_full"}
            )
            ablation_df = load_seed_nse(
                ablation_test_dir / "metrics.json",
                gage_ids,
            ).rename(columns={"nse": "nse_ablation"})
            merged = full_df.merge(
                ablation_df[["gage_id", "basin_idx", "nse_ablation"]],
                on=["gage_id", "basin_idx"],
                how="inner",
                validate="one_to_one",
            )
            if len(merged) != len(gage_ids):
                raise ValueError(
                    f"Seed {seed} / {ablation_name} merged to {len(merged)} basins; "
                    f"expected {len(gage_ids)}."
                )
            merged["process"] = process
            merged["ablation_name"] = ablation_name
            merged["seed"] = seed
            merged["delta_nse"] = merged["nse_full"] - merged["nse_ablation"]
            rows.append(
                merged[
                    [
                        "process",
                        "ablation_name",
                        "seed",
                        "basin_idx",
                        "gage_id",
                        "nse_full",
                        "nse_ablation",
                        "delta_nse",
                    ]
                ]
            )
    return pd.concat(rows, ignore_index=True), source_paths


def aggregate_basin_level(seed_df: pd.DataFrame) -> pd.DataFrame:
    grouped = (
        seed_df.groupby(["process", "ablation_name", "basin_idx", "gage_id"], as_index=False)
        .agg(
            n_seeds_full=("nse_full", lambda s: int(np.isfinite(pd.to_numeric(s, errors="coerce")).sum())),
            n_seeds_ablation=("nse_ablation", lambda s: int(np.isfinite(pd.to_numeric(s, errors="coerce")).sum())),
            nse_full_median=("nse_full", "median"),
            nse_ablation_median=("nse_ablation", "median"),
            delta_nse_median=("delta_nse", "median"),
            delta_nse_min=("delta_nse", "min"),
            delta_nse_max=("delta_nse", "max"),
            delta_nse_std=("delta_nse", lambda s: float(pd.Series(s, dtype=float).std(ddof=0))),
        )
    )
    return grouped.sort_values(["process", "gage_id"]).reset_index(drop=True)


def summarize_processes(basin_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for process, sub in basin_df.groupby("process", sort=True):
        delta = pd.to_numeric(sub["delta_nse_median"], errors="coerce")
        rows.append(
            {
                "process": process,
                "n_basins_total": int(len(sub)),
                "n_basins_degraded": int((delta > 0).sum()),
                "fraction_degraded": float((delta > 0).mean()),
                "mean_delta_nse": float(delta.mean()),
                "median_delta_nse": float(delta.median()),
            }
        )
    return pd.DataFrame(rows).sort_values("process").reset_index(drop=True)


def write_summary_markdown(path: Path, summary_df: pd.DataFrame) -> None:
    lines = [
        "# Full-MOPEX LOPO summary",
        "",
        "| process | degraded basins | fraction degraded | mean delta NSE | median delta NSE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_df.itertuples(index=False):
        lines.append(
            f"| {row.process} | {row.n_basins_degraded} / {row.n_basins_total} | "
            f"{row.fraction_degraded:.4f} | {row.mean_delta_nse:.6f} | "
            f"{row.median_delta_nse:.6f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_model_config_for_audit(
    *,
    config_path: str,
    output_root: Path,
    run_name: str,
    model_type: str,
    seed: int,
    fixed_weights: list[float] | None = None,
) -> dict[str, Any]:
    import argparse as _argparse

    from project.flexmopex import load_config
    from project.flexmopex.run_model import _resolve_config, apply_runtime_overrides

    config_path_str = _resolve_config(config_path)
    config = load_config(config_path_str)
    fake_args = _argparse.Namespace(
        alpha_pos=None,
        config=config_path,
        alpha=0.0,
        mode="test",
        seed=seed,
        gpu_id=0,
        test_epoch=50,
        start_epoch=None,
        epochs=None,
        batch_size=None,
        nmul=None,
        fixed_weights=fixed_weights,
        output_root=str(output_root),
        run_name=run_name,
        preflight_only=False,
        verbose=False,
        model_type=model_type,
        loro_holdout_region=None,
    )
    apply_runtime_overrides(config, fake_args, config_path=config_path_str)
    return config


def _state_l1_difference(full_state: dict[str, torch.Tensor], ablation_state: dict[str, torch.Tensor]) -> float:
    total = 0.0
    shared_keys = sorted(set(full_state) & set(ablation_state))
    for key in shared_keys:
        full_tensor = full_state[key]
        ablation_tensor = ablation_state[key]
        if full_tensor.shape != ablation_tensor.shape:
            continue
        total += float(torch.sum(torch.abs(full_tensor - ablation_tensor)).item())
    return total


def _move_dataset_to_device(
    dataset: dict[str, Any],
    device: str | torch.device,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in dataset.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def write_retraining_audit(
    path: Path,
    *,
    root: Path,
    full_root: Path,
    seed: int,
    process: str,
    ablation_name: str,
) -> None:
    from project.flexmopex.local_model_handler import FlexMopexModelHandler
    from project.flexmopex.run_model import _build_data_loader

    full_run_name = f"config_dmopex_v1/full_alpha_0/seed_{seed}"
    ablation_run_name = f"config_dmopex_v1/{ablation_name}/seed_{seed}"
    config_path = "conf/config_dmopex_v1.yaml"
    full_output_root = full_root / f"seed{seed}"

    full_config = _load_model_config_for_audit(
        config_path=config_path,
        output_root=full_output_root,
        run_name=full_run_name,
        model_type="full",
        seed=seed,
    )
    ablation_config = _load_model_config_for_audit(
        config_path=config_path,
        output_root=root,
        run_name=ablation_run_name,
        model_type="fixed",
        seed=seed,
        fixed_weights={
            "phenology": [0.0, 1.0, 1.0, 1.0],
            "interception": [1.0, 0.0, 1.0, 1.0],
            "snow": [1.0, 1.0, 0.0, 1.0],
            "subsurface": [1.0, 1.0, 1.0, 0.0],
        }[process],
    )

    full_loader = _build_data_loader(full_config)
    ablation_loader = _build_data_loader(ablation_config)
    full_model = FlexMopexModelHandler(full_config, verbose=False)
    ablation_model = FlexMopexModelHandler(ablation_config, verbose=False)
    full_model.load_model(full_config["test"]["test_epoch"])
    ablation_model.load_model(ablation_config["test"]["test_epoch"])

    full_inner = next(iter(full_model.model_dict.values()))
    ablation_inner = next(iter(ablation_model.model_dict.values()))
    full_eval = _move_dataset_to_device(
        full_inner._ensure_static_features(full_loader.eval_dataset),
        full_inner.device,
    )
    ablation_eval = _move_dataset_to_device(
        ablation_inner._ensure_static_features(ablation_loader.eval_dataset),
        ablation_inner.device,
    )
    with torch.no_grad():
        full_params = full_inner.nn_model(full_eval)
        ablation_params = ablation_inner.nn_model(ablation_eval)

    param_diff = float(
        torch.mean(torch.abs(full_params["params"] - ablation_params["params"])).item()
    )
    gamma_diff = float(
        torch.mean(torch.abs(full_params["gamma_uh"] - ablation_params["gamma_uh"])).item()
    )
    state_l1 = _state_l1_difference(
        full_inner.nn_model.state_dict(),
        ablation_inner.nn_model.state_dict(),
    )
    if max(param_diff, gamma_diff, state_l1) <= 1e-6:
        raise ValueError(
            "Retraining audit failed: parameter outputs and NN checkpoint tensors "
            "did not diverge from Full."
        )

    lines = [
        "# Retraining audit",
        "",
        f"- Seed: `{seed}`",
        f"- Ablation: `{ablation_name}`",
        f"- Process removed: `{process}`",
        f"- Mean abs diff in predicted hydrologic parameters: `{param_diff:.6e}`",
        f"- Mean abs diff in predicted routing parameters: `{gamma_diff:.6e}`",
        f"- L1 diff across NN checkpoint tensors: `{state_l1:.6e}`",
        "",
        "Interpretation: fixed structural masks remain constant by design; "
        "nonzero differences above confirm the parameter head reoptimized relative to Full.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    full_root = Path(args.full_root).resolve()
    analysis_dir = root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    seed_df, _ = collect_seed_level(root, full_root, args.seeds)
    basin_df = aggregate_basin_level(seed_df)
    summary_df = summarize_processes(basin_df)

    seed_path = analysis_dir / "lopo_seed_level.csv"
    basin_path = analysis_dir / "lopo_basin_level.csv"
    summary_path = analysis_dir / "lopo_process_summary.csv"
    markdown_path = analysis_dir / "lopo_process_summary.md"
    audit_path = analysis_dir / "retraining_audit.md"

    seed_df.to_csv(seed_path, index=False)
    basin_df.to_csv(basin_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    write_summary_markdown(markdown_path, summary_df)
    write_retraining_audit(
        audit_path,
        root=root,
        full_root=full_root,
        seed=args.seeds[0],
        process="snow",
        ablation_name=PROCESS_SPECS["snow"],
    )

    print(f"Wrote {seed_path}")
    print(f"Wrote {basin_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {markdown_path}")
    print(f"Wrote {audit_path}")


if __name__ == "__main__":
    main()
