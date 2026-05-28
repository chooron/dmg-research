#!/usr/bin/env python
"""Entry point for parameter learning experiments.

Usage
-----
python run_parameter_learning.py \\
    --model-id hbv96 \\
    --objective KGE \\
    --config conf/param_learning_kge.yaml \\
    --device cuda:0 \\
    --epochs 100 \\
    --seeds 42 123 456 789 1234
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from benchmark.parameter_learning import ParameterLearningConfig, run_parameter_learning


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Parameter learning for dual-evidence benchmark")
    p.add_argument("--model-id", required=True, help="Model ID (e.g. hbv96, m01, ...)")
    p.add_argument("--objective", required=True, choices=["KGE", "KGE_LOG", "NSE", "LOG_NSE"],
                   help="Training objective")
    p.add_argument("--config", default="conf/param_learning_kge.yaml",
                   help="Path to YAML config file")
    p.add_argument("--device", default="cpu", help="PyTorch device (cpu, cuda:0, ...)")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs from config")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Override random seeds from config")
    p.add_argument("--hidden-size", type=int, default=None,
                   help="Override MLP hidden size from config")
    p.add_argument("--output-dir", default=None,
                   help="Override output directory from config")
    p.add_argument("--basin-ids-path", default=None,
                   help="Override basin IDs file path")
    p.add_argument("--data-root", default=None,
                   help="Override data root directory")
    p.add_argument("--attributes-path", default=None,
                   help="Override attributes file path")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate")
    return p.parse_args()


def load_yaml_config(config_path: str) -> dict:
    """Load YAML config file."""
    try:
        import yaml
    except ImportError:
        print("Warning: PyYAML not installed, using defaults only")
        return {}
    p = Path(config_path)
    if not p.exists():
        print(f"Warning: config file {config_path} not found, using defaults")
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args = parse_args()
    yaml_cfg = load_yaml_config(args.config)

    # Build ParameterLearningConfig from YAML + CLI overrides
    train_cfg = yaml_cfg.get("train", {})
    model_cfg = yaml_cfg.get("model", {})
    paths_cfg = yaml_cfg.get("paths", {})

    # Load the full benchmark config to pass to CamelsStore
    from benchmark.config import load_benchmark_config
    benchmark_config_path = paths_cfg.get(
        "benchmark_config",
        str(Path(__file__).parent / "conf" / "benchmark_kge.yaml")
    )
    try:
        full_benchmark_config = load_benchmark_config(benchmark_config_path)
    except Exception as e:
        print(f"Warning: could not load benchmark config from {benchmark_config_path}: {e}")
        full_benchmark_config = {}

    # Override device in benchmark config
    if full_benchmark_config.get("calibration"):
        full_benchmark_config["calibration"]["device"] = args.device

    cfg = ParameterLearningConfig(
        model_id=args.model_id,
        objective=args.objective,
        benchmark_config=full_benchmark_config,
        basin_ids_path=args.basin_ids_path or paths_cfg.get("basin_ids_path", "data/559sub_id.txt"),
        data_root=args.data_root or paths_cfg.get("data_root", ""),
        attributes_path=args.attributes_path or paths_cfg.get("attributes_path", ""),
        train_start=yaml_cfg.get("splits", {}).get("train", {}).get("start_time", "1989-01-01"),
        train_end=yaml_cfg.get("splits", {}).get("train", {}).get("end_time", "1998-12-31"),
        test_start=yaml_cfg.get("splits", {}).get("test", {}).get("start_time", "1999-01-01"),
        test_end=yaml_cfg.get("splits", {}).get("test", {}).get("end_time", "2009-12-31"),
        warmup_days=model_cfg.get("warm_up", 365),
        epochs=args.epochs or train_cfg.get("epochs", 100),
        lr=args.lr or train_cfg.get("lr", 1e-3),
        seeds=args.seeds or train_cfg.get("seeds", [42, 123, 456, 789, 1234]),
        hidden_size=args.hidden_size or model_cfg.get("nn", {}).get("hidden_size", 128),
        kge_log_eps_frac=train_cfg.get("kge_log_eps_frac", 0.01),
        kge_log_global_eps=train_cfg.get("kge_log_global_eps", 1e-3),
        output_dir=args.output_dir or paths_cfg.get("output_dir", "outputs/parameter_learning"),
        device=args.device,
        log_interval=train_cfg.get("log_interval", 10),
    )

    run_parameter_learning(cfg)


if __name__ == "__main__":
    main()
