"""run_batch_fixed_weights.py - Run one fixed-mask experiment across multiple seeds.

This mirrors the batch LORO runner pattern: one Python process handles one
ablation across all requested seeds so torch.compile caches can be reused.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torch_inductor_cache")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")

import torch  # noqa: E402

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
DATA_ROOT = REPO_ROOT / "data"
os.environ.setdefault("FLEXMOPEX_DATA_DIR", str(DATA_ROOT))
os.environ.setdefault("DATA_PATH", str(DATA_ROOT))
os.environ.setdefault("BASIN_GROUPS_DIR", str(DATA_ROOT / "basin_groups"))
os.environ.setdefault("GAGE_INFO", str(DATA_ROOT / "gage_id.npy"))
for path in (REPO_ROOT, PROJECT_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from project.flexmopex import load_config  # noqa: E402
from project.flexmopex.run_model import (  # noqa: E402
    _resolve_config,
    apply_runtime_overrides,
    run_test,
    run_train,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch fixed-weight runner")
    parser.add_argument("--ablation-name", required=True, help="Output name for this ablation.")
    parser.add_argument(
        "--fixed-weights",
        type=float,
        nargs=4,
        required=True,
        metavar=("W_PHEN", "W_INT", "W_SNOW", "W_SUB"),
        help="Fixed weights in strict order: w_phen w_int w_snow w_sub.",
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--gpu-id", type=int, default=0, dest="gpu_id")
    parser.add_argument("--config", default="conf/config_dmopex_v1.yaml")
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_DIR / "results" / "block1_full_lopo"),
        dest="output_root",
    )
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--mode",
        choices=("train", "test", "train_test"),
        default="train_test",
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    return parser.parse_args(argv)


def _make_fake_args(args: argparse.Namespace, seed: int) -> argparse.Namespace:
    import argparse as _argparse

    return _argparse.Namespace(
        alpha_pos=None,
        config=args.config,
        alpha=args.alpha,
        mode=args.mode,
        seed=seed,
        gpu_id=args.gpu_id,
        test_epoch=None,
        start_epoch=None,
        epochs=args.epochs,
        batch_size=None,
        nmul=None,
        fixed_weights=list(args.fixed_weights),
        output_root=args.output_root,
        run_name=f"config_dmopex_v1/{args.ablation_name}/seed_{seed}",
        preflight_only=False,
        verbose=args.verbose,
        model_type="fixed",
        loro_holdout_region=None,
    )


def _is_complete(output_root: str, ablation_name: str, seed: int) -> bool:
    base_dir = Path(output_root) / "config_dmopex_v1" / ablation_name / f"seed_{seed}"
    return any(base_dir.rglob("test*_Ep*/metrics_agg.json"))


def run_one_seed(args: argparse.Namespace, *, config_path_str: str, seed: int) -> bool:
    tag = f"{args.ablation_name}_seed{seed}"
    if _is_complete(args.output_root, args.ablation_name, seed):
        print(f"[{tag}] SKIP (already complete)")
        return True
    print(f"\n{'=' * 60}")
    print(f"[{tag}] Starting  (gpu={args.gpu_id})")
    t0 = time.perf_counter()
    try:
        config = load_config(config_path_str)
        fake_args = _make_fake_args(args, seed)
        apply_runtime_overrides(config, fake_args, config_path=config_path_str)
        if args.mode in {"train", "train_test"}:
            run_train(config, args.verbose)
        if args.mode in {"test", "train_test"}:
            run_test(config, args.verbose)
        elapsed = time.perf_counter() - t0
        print(f"[{tag}] Done  ({elapsed:.1f}s)")
        return True
    except Exception as exc:  # noqa: BLE001
        elapsed = time.perf_counter() - t0
        print(f"[{tag}] FAILED after {elapsed:.1f}s: {exc}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        return False
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config_path_str = _resolve_config(args.config)
    print(
        f"Batch fixed-weight run | ablation={args.ablation_name} | "
        f"seeds={args.seeds} | fixed_weights={list(args.fixed_weights)}"
    )
    print(f"TORCHINDUCTOR_CACHE_DIR={os.environ.get('TORCHINDUCTOR_CACHE_DIR')}")
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    results: dict[str, bool] = {}
    for seed in args.seeds:
        tag = f"{args.ablation_name}_seed{seed}"
        results[tag] = run_one_seed(args, config_path_str=config_path_str, seed=seed)

    print(f"\n{'=' * 60}")
    n_ok = sum(results.values())
    for tag, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {status}  {tag}")
    print(f"{n_ok}/{len(results)} succeeded")
    if n_ok < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
