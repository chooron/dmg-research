"""run_batch_loro.py – Run all (seed, model_type) combos for ONE region in a single process.

Optimization rationale:
- torch.compile is called once per model_type and reused across seeds via cache.
- A persistent TORCHINDUCTOR_CACHE_DIR lets even separate processes share kernels.
- Running all combos in one process avoids N×M cold-start compile penalties.

Usage:
    python run_batch_loro.py --region 0 --seeds 42 123 456 \
        --model-types flex full base --gpu-id 0

Environment variables forwarded to run_model internals:
    TORCHINDUCTOR_CACHE_DIR  (default: /tmp/torch_inductor_cache)
    DATA_PATH, BASIN_GROUPS_DIR, etc.
"""
from __future__ import annotations

import argparse
import copy
import gc
import os
import sys
import time
from pathlib import Path
from typing import Any

# ── Persistent compile cache (set before any torch import) ──────────────────
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/tmp/torch_inductor_cache")
os.environ.setdefault("TORCH_COMPILE_DEBUG", "0")

import torch  # noqa: E402  (must come after env-var setup)

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
for p in (REPO_ROOT, PROJECT_DIR):
    s = str(p)
    if s not in sys.path:
        sys.path.insert(0, s)

# Import after path setup
from project.flexmopex.run_model import (  # noqa: E402
    apply_runtime_overrides,
    run_loro_train,
    _resolve_config,
)
from project.flexmopex import load_config  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch LORO runner (one process per region)")
    p.add_argument("--region", type=int, required=True, help="Holdout region index (0-6)")
    p.add_argument(
        "--seeds", type=int, nargs="+", default=[42], help="Random seeds to run"
    )
    p.add_argument(
        "--model-types",
        nargs="+",
        default=["flex", "full", "base"],
        choices=["flex", "full", "base"],
        dest="model_types",
    )
    p.add_argument("--gpu-id", type=int, default=0, dest="gpu_id")
    p.add_argument(
        "--config",
        default="conf/config_dmopex_v1.yaml",
        help="Config path relative to project/flexmopex",
    )
    p.add_argument(
        "--output-root",
        default=str(PROJECT_DIR / "results" / "block3_loro"),
        dest="output_root",
    )
    p.add_argument("--alpha", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--verbose", action="store_true", default=False)
    return p.parse_args(argv)


def _make_fake_args(
    region: int,
    seed: int,
    model_type: str,
    gpu_id: int,
    config: str,
    output_root: str,
    alpha: float,
    epochs: int | None,
    verbose: bool,
) -> argparse.Namespace:
    """Build an argparse.Namespace compatible with apply_runtime_overrides."""
    import argparse as _ap

    ns = _ap.Namespace(
        alpha_pos=None,
        config=config,
        alpha=alpha,
        mode="train",
        seed=seed,
        gpu_id=gpu_id,
        test_epoch=None,
        start_epoch=None,
        epochs=epochs,
        batch_size=None,
        nmul=None,
        output_root=output_root,
        run_name=None,
        preflight_only=False,
        verbose=verbose,
        model_type=model_type,
        loro_holdout_region=region,
    )
    return ns


def run_one(
    region: int,
    seed: int,
    model_type: str,
    *,
    gpu_id: int,
    config_path_str: str,
    output_root: str,
    alpha: float,
    epochs: int | None,
    verbose: bool,
) -> bool:
    tag = f"{model_type}_region{region}_seed{seed}"
    print(f"\n{'='*60}")
    print(f"[{tag}] Starting  (gpu={gpu_id})")
    t0 = time.perf_counter()

    try:
        # Fresh config dict for each experiment
        config = load_config(config_path_str)
        args = _make_fake_args(
            region=region,
            seed=seed,
            model_type=model_type,
            gpu_id=gpu_id,
            config=config_path_str,
            output_root=output_root,
            alpha=alpha,
            epochs=epochs,
            verbose=verbose,
        )
        apply_runtime_overrides(config, args, config_path=config_path_str)
        run_loro_train(config, verbose)
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
        # Release GPU memory between experiments
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    config_path_str = _resolve_config(args.config)
    region = args.region
    seeds = args.seeds
    model_types = args.model_types
    total = len(seeds) * len(model_types)

    print(f"Batch LORO | region={region} | seeds={seeds} | model_types={model_types}")
    print(f"Total experiments: {total}")
    print(f"TORCHINDUCTOR_CACHE_DIR={os.environ.get('TORCHINDUCTOR_CACHE_DIR')}")
    print(f"GPU: cuda:{args.gpu_id}")

    # Set GPU device once for the whole process
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)

    results: dict[str, bool] = {}
    for seed in seeds:
        for model_type in model_types:
            tag = f"{model_type}_region{region}_seed{seed}"
            ok = run_one(
                region=region,
                seed=seed,
                model_type=model_type,
                gpu_id=args.gpu_id,
                config_path_str=config_path_str,
                output_root=args.output_root,
                alpha=args.alpha,
                epochs=args.epochs,
                verbose=args.verbose,
            )
            results[tag] = ok

    print(f"\n{'='*60}")
    print(f"Batch complete for region {region}:")
    n_ok = sum(results.values())
    for tag, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {status}  {tag}")
    print(f"{n_ok}/{total} succeeded")
    if n_ok < total:
        sys.exit(1)


if __name__ == "__main__":
    main()
