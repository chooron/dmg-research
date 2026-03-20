"""
train_causal_dpl.py
-------------------
Main entry point for Causal-dPL effective-cluster holdout cross-validation.

Usage
-----
    python train_causal_dpl.py --config conf/config_dhbv.yaml \
                                --holdout A          # hold out effective cluster A
                                --mode train_test    # train + OOD eval

    # Run all 7 effective-cluster holdouts sequentially:
    for c in A B C D E F G; do
        python train_causal_dpl.py --config conf/config_dhbv.yaml --holdout $c
    done

Config extensions (add to your yaml under a 'causal:' key)
-----------------------------------------------------------
causal:
  cluster_csv:    data/gauge_pos_cluster_climate.csv
  basin_ids_path: data/gage_id.txt          # or gage_id.npy
  holdout_cluster: A                        # overridden by --holdout CLI arg

lambda_irm:          100.0
irm_warmup_epochs:   10
"""

import argparse
import logging
import os
import sys

import torch

# Allow running from repo root
sys.path.insert(0, os.path.dirname(__file__))

from dmg.core.data.loaders import HydroLoader
from dmg.core.utils import set_randomseed, print_config
from omegaconf import OmegaConf
from dmg.core.utils.utils import initialize_config

from implements import build_causal_dpl, CausalTrainer
from implements.gnann_splitter import GnannEnvironmentSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('train_causal_dpl')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Causal-dPL training / evaluation')
    p.add_argument('--config',   default='conf/config_irm_dhbv.yaml',
                   help='Path to dmg config yaml')
    p.add_argument('--holdout',  type=str, default=None,
                   help='Effective cluster to hold out for OOD eval (A-G). '
                        'Overrides causal.holdout_cluster in config.')
    p.add_argument('--mode',     default=None,
                   help='Override config mode: train | test | train_test')
    p.add_argument('--lambda-irm', type=float, default=None,
                   help='Override lambda_irm in config')
    p.add_argument('--epochs',   type=int, default=None,
                   help='Override train.epochs in config')
    p.add_argument('--seed',     type=int, default=None,
                   help='Override experiment seed')
    return p.parse_args()


def _get_penalty_warmup_epochs(config: dict) -> int:
    loss_name = config['train']['loss_function']['name']
    if loss_name == 'IRMKgeBatchLoss':
        return int(config.get('irm_warmup_epochs', 0) or 0)
    if loss_name == 'VRExKgeBatchLoss':
        return int(config.get('vrex_warmup_epochs', 0) or 0)
    return 0



def _build_optimizer_and_scheduler(config: dict, model: torch.nn.Module):
    optimizer_name = config['train'].get('optimizer', {}).get('name', 'Adam')
    if optimizer_name != 'Adam':
        raise ValueError(f"Unsupported optimizer '{optimizer_name}'. Only Adam is supported.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config['train']['lr'],
    )

    lr_cfg = config['train'].get('lr_scheduler', {})
    scheduler_name = lr_cfg.get('name', 'CosineAnnealingLR') if isinstance(lr_cfg, dict) else lr_cfg
    if scheduler_name != 'CosineAnnealingLR':
        log.info("Skipping custom scheduler wiring for unsupported scheduler '%s'.", scheduler_name)
        return optimizer, None

    total_epochs = int(config['train']['epochs'])
    warm_epochs = _get_penalty_warmup_epochs(config)
    eta_min = lr_cfg.get('eta_min', 1e-5) if isinstance(lr_cfg, dict) else 1e-5

    if 0 < warm_epochs < total_epochs and (warm_epochs / total_epochs) >= 0.05:
        start_factor = (
            lr_cfg.get('warmup_start_factor', 0.1) if isinstance(lr_cfg, dict) else 0.1
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(
                    optimizer,
                    start_factor=start_factor,
                    end_factor=1.0,
                    total_iters=warm_epochs,
                ),
                torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=max(total_epochs - warm_epochs, 1),
                    eta_min=eta_min,
                ),
            ],
            milestones=[warm_epochs],
        )
        log.info(
            "Using LinearLR + CosineAnnealingLR (%d warmup epochs, %d cosine epochs).",
            warm_epochs,
            total_epochs - warm_epochs,
        )
        return optimizer, scheduler

    if warm_epochs > 0:
        log.info(
            "Penalty warmup is short (%d/%d epochs); keeping plain CosineAnnealingLR.",
            warm_epochs,
            total_epochs,
        )

    t_max = lr_cfg.get('T_max', total_epochs) if isinstance(lr_cfg, dict) else total_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=eta_min,
    )
    return optimizer, scheduler


def _load_basin_ids(path: str):
    import numpy as np

    if path.endswith('.npy'):
        return np.load(path, allow_pickle=True).astype(int)
    return np.loadtxt(path, dtype=int)


def _log_effective_cluster_fold_sizes(causal_cfg: dict) -> None:
    splitter = GnannEnvironmentSplitter(
        cluster_csv=causal_cfg['cluster_csv'],
        basin_ids=_load_basin_ids(causal_cfg['basin_ids_path']),
        holdout_cluster=causal_cfg.get('holdout_cluster'),
    )
    if len(splitter.unassigned_indices) > 0:
        log.info(
            "Excluded %d basins without effective-cluster labels from leave-one-cluster splits.",
            len(splitter.unassigned_indices),
        )
    for fold in splitter.fold_size_summary():
        status = "OK" if fold['matches_expected'] else "MISMATCH"
        log.info(
            "Fold %s | test=%d | train=%d | train_envs=%d | %s",
            fold['held_out_cluster'],
            fold['test_basins'],
            fold['train_basins'],
            fold['train_environments'],
            status,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Load config and apply CLI overrides before interpolation resolves.
    raw_config = OmegaConf.load(args.config)
    original_epochs = raw_config['train']['epochs']

    if args.mode:
        raw_config['mode'] = args.mode
    if args.holdout is not None:
        if raw_config.get('causal') is None:
            raw_config['causal'] = {}
        raw_config['causal']['holdout_cluster'] = args.holdout
    if args.lambda_irm is not None:
        raw_config['lambda_irm'] = args.lambda_irm
    if args.epochs is not None:
        raw_config['train']['epochs'] = args.epochs
        lr_cfg = raw_config['train'].get('lr_scheduler')
        if (
            isinstance(lr_cfg, dict)
            and lr_cfg.get('name') == 'CosineAnnealingLR'
            and ('T_max' not in lr_cfg or lr_cfg.get('T_max') == original_epochs)
        ):
            lr_cfg['T_max'] = args.epochs
    if args.seed is not None:
        raw_config['seed'] = args.seed

    config = initialize_config(raw_config)

    # Validate causal config block
    causal_cfg = config.get('causal', {})
    for required in ('cluster_csv', 'basin_ids_path'):
        if required not in causal_cfg:
            raise ValueError(
                f"Missing 'causal.{required}' in config. "
                "Add a 'causal:' block to your yaml (see module docstring)."
            )

    holdout_cluster = causal_cfg.get('holdout_cluster')
    _log_effective_cluster_fold_sizes(causal_cfg)
    log.info(
        "Mode: %s  |  Held-out effective cluster: %s  |  Model dir: %s",
        config['mode'],
        holdout_cluster,
        config['model_dir'],
    )
    # Flatten lr_scheduler dict to name string only for print_config, then restore
    lr_sched = config['train'].get('lr_scheduler')
    if isinstance(lr_sched, dict):
        config['train']['lr_scheduler'] = lr_sched.get('name', str(lr_sched))
    print_config(config)
    if isinstance(lr_sched, dict):
        config['train']['lr_scheduler'] = lr_sched

    set_randomseed(config['seed'])

    # 2. Load data
    log.info("Loading datasets...")
    data_loader = HydroLoader(config, test_split=True, overwrite=False)

    # 3. Build model
    log.info("Building CausalDplModel (HbvStatic + AnnModel)...")
    model = build_causal_dpl(config)
    model = model.to(config['device'])
    log.info(f"Model device: {config['device']}")
    optimizer, scheduler = _build_optimizer_and_scheduler(config, model)

    # 4. Build trainer
    trainer = CausalTrainer(
        config,
        model=model,
        train_dataset=data_loader.train_dataset,
        eval_dataset=data_loader.eval_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        write_out=True,
        verbose=False,
    )

    # 5. Run
    mode = config['mode']

    if 'train' in mode:
        log.info("Starting fixed-epoch causal training...")
        trainer.train()
        log.info(f"Training complete. Model saved to {config['model_dir']}")

    if 'test' in mode or mode == 'train_test':
        if holdout_cluster is not None:
            log.info(f"Evaluating on held-out effective cluster {holdout_cluster}...")
            trainer.evaluate_holdout()
        else:
            log.info("Evaluating on full eval dataset (no holdout set)...")
            trainer.evaluate()
        log.info(f"Metrics saved to {config['output_dir']}")


if __name__ == '__main__':
    main()
