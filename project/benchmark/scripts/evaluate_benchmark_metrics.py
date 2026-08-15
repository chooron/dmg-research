#!/usr/bin/env python3
"""
Unified Benchmark Evaluation Script for 36 Hydrological Models.
Evaluates frozen model parameters on both Train and Test (Validation) periods
across 531 CAMELS basins, computing KGE scores, deltas, and win rates vs MARRMoT baselines.
"""
from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

BENCHMARK_ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(BENCHMARK_ROOT), str(BENCHMARK_ROOT / "src")]

from src.checkpoint_guard import validate_canonical_checkpoint
from src.checkpointing import load_checkpoint
from src.data_selection import evaluate_period, frozen_parameters, load_ids
from src.model_registry import NPARAM_INFO_36
from src.production_config import load_resolved_config, validate_full_run_config


def median_or_nan(values: pd.Series) -> float:
    return float(values.median()) if values.notna().any() else float("nan")


def load_marrmot_scores(marrmot_dir: Path, model_name: str) -> pd.DataFrame:
    """Load reference MARRMoT scores if available."""
    ref_path = marrmot_dir / f"{model_name}.csv"
    if not ref_path.is_file():
        return pd.DataFrame({"basin_id": [], "marrmot_train_kge": [], "marrmot_test_kge": []})
    df = pd.read_csv(ref_path)
    return df.rename(columns={"train_kge": "marrmot_train_kge", "test_kge": "marrmot_test_kge"})


def evaluate_single_model(model_dir: Path, config: dict, device: str, backend: str, marrmot_dir: Path) -> tuple[pd.DataFrame, dict]:
    model_name = model_dir.name
    starts = int(config["optimization"]["starts"])
    generations = int(config["optimization"]["generations"])
    # Canonical provenance guard: reject pilot / intermediate (gen < required)
    # checkpoints loudly instead of silently evaluating them.
    validate_canonical_checkpoint(
        model_dir, model_name=model_name,
        required_generation=generations, required_basins=531,
    )

    basin_ids, latent, checkpoint_train = frozen_parameters(model_dir, generations, starts)
    train_kge = evaluate_period(model_name, latent, basin_ids, config, "train", device, backend)
    test_kge = evaluate_period(model_name, latent, basin_ids, config, "test", device, backend)

    frame = pd.DataFrame({
        "model": model_name,
        "basin_id": basin_ids,
        "selected_checkpoint_train_kge": checkpoint_train,
        "train_kge": train_kge,
        "test_kge": test_kge,
    })

    marrmot_df = load_marrmot_scores(marrmot_dir, model_name)
    if not marrmot_df.empty:
        frame = frame.merge(marrmot_df, on="basin_id", how="left")
        common = frame.dropna(subset=["marrmot_train_kge", "marrmot_test_kge"]).copy()
        common["train_delta_vs_marrmot"] = common["train_kge"] - common["marrmot_train_kge"]
        common["test_delta_vs_marrmot"] = common["test_kge"] - common["marrmot_test_kge"]
    else:
        common = pd.DataFrame()

    summary_row = {
        "model": model_name,
        "n_basins": len(frame),
        "n_common_marrmot": len(common),
        "starts": starts,
        "generation": generations,
        "selection": "best_of_10_checkpoint_train_kge_only",
        "train_kge_median": median_or_nan(frame["train_kge"]),
        "test_kge_median": median_or_nan(frame["test_kge"]),
        "marrmot_train_kge_median_common": median_or_nan(common["marrmot_train_kge"]) if not common.empty else float("nan"),
        "marrmot_test_kge_median_common": median_or_nan(common["marrmot_test_kge"]) if not common.empty else float("nan"),
        "train_kge_median_common": median_or_nan(common["train_kge"]) if not common.empty else float("nan"),
        "test_kge_median_common": median_or_nan(common["test_kge"]) if not common.empty else float("nan"),
        "paired_train_delta_median": median_or_nan(common["train_delta_vs_marrmot"]) if not common.empty else float("nan"),
        "paired_test_delta_median": median_or_nan(common["test_delta_vs_marrmot"]) if not common.empty else float("nan"),
        "train_win_fraction": float((common["train_delta_vs_marrmot"] >= 0).mean()) if not common.empty else float("nan"),
        "test_win_fraction": float((common["test_delta_vs_marrmot"] >= 0).mean()) if not common.empty else float("nan"),
    }
    return frame, summary_row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate 36 Hydrological Models on Train & Test Sets")
    parser.add_argument("--checkpoint-root", required=True, help="Directory containing completed model checkpoint subfolders")
    parser.add_argument("--config", default="configs/full_run_10starts_300gen_warm1980_1981x5.yaml")
    parser.add_argument("--marrmot-dir", default="references/marrmot_obj1")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--backend", choices=["eager", "compile"], default="compile")
    args = parser.parse_args()

    ckpt_root = Path(args.checkpoint_root) if Path(args.checkpoint_root).is_absolute() else BENCHMARK_ROOT / args.checkpoint_root
    config_path = Path(args.config) if Path(args.config).is_absolute() else BENCHMARK_ROOT / args.config
    out_dir = Path(args.output_dir) if Path(args.output_dir).is_absolute() else BENCHMARK_ROOT / args.output_dir
    marrmot_path = Path(args.marrmot_dir) if Path(args.marrmot_dir).is_absolute() else BENCHMARK_ROOT / args.marrmot_dir

    config = load_resolved_config(config_path)
    validate_full_run_config(config)
    out_dir.mkdir(parents=True, exist_ok=True)

    completed_dirs = [p for p in sorted(ckpt_root.iterdir()) if p.is_dir() and (p / "DONE").is_file()]
    if not completed_dirs:
        raise RuntimeError(f"No completed model directories found in {ckpt_root}")

    print(f"=== Evaluating {len(completed_dirs)} completed models ===")
    all_frames, summaries, failures = [], [], []

    for idx, mdir in enumerate(completed_dirs, 1):
        t0 = time.perf_counter()
        try:
            frame, summary = evaluate_single_model(mdir, config, args.device, args.backend, marrmot_path)
            summary["elapsed_s"] = time.perf_counter() - t0
            all_frames.append(frame)
            summaries.append(summary)
            print(f"[{idx}/{len(completed_dirs)}] Evaluated [{mdir.name}] in {summary['elapsed_s']:.2f}s | Train KGE: {summary['train_kge_median']:.4f} | Test KGE: {summary['test_kge_median']:.4f}")
        except Exception as exc:
            failures.append({"model": mdir.name, "error": str(exc)})
            print(f"[{idx}/{len(completed_dirs)}] FAILED [{mdir.name}]: {exc}")
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    by_basin = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    by_model = pd.DataFrame(summaries).sort_values("model") if summaries else pd.DataFrame()

    by_basin.to_csv(out_dir / "full300_kge_by_basin.csv", index=False)
    by_model.to_csv(out_dir / "full300_kge_model_summary.csv", index=False)
    if failures:
        pd.DataFrame(failures).to_csv(out_dir / "full300_kge_evaluation_failures.csv", index=False)

    overall = {
        "selection_rule": "best_of_10_by_train_kge_only; test KGE evaluated after parameters are frozen",
        "models_evaluated": len(summaries),
        "models_failed": len(failures),
        "model_median_train_kge_median": median_or_nan(by_model["train_kge_median"]) if not by_model.empty else float("nan"),
        "model_median_test_kge_median": median_or_nan(by_model["test_kge_median"]) if not by_model.empty else float("nan"),
    }
    (out_dir / "full300_kge_overall.json").write_text(json.dumps(overall, indent=2) + "\n")
    print(f"\n=== Overall Evaluation Summary ===")
    print(json.dumps(overall, indent=2))


if __name__ == "__main__":
    main()
