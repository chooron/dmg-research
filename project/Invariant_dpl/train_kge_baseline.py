"""
train_kge_baseline.py
---------------------
Baseline ERM training / evaluation for the static-attribute dPL setup.

This entry point is aligned with ``train_causal_dpl.py`` in terms of:

- ``McMlpModel`` + ``HbvStatic`` interface checks,
- ``nmul=1`` static-parameter setup,
- leave-one-effective-cluster holdout evaluation,
- MC-dropout evaluation and ``metrics_avg`` selection artifacts.

The main difference is the training loss:

- baseline: ``KgeBatchLoss``
- causal: ``IRMKgeBatchLoss`` / ``VRExKgeBatchLoss``
"""

import argparse
import logging
import os
import sys
from pathlib import Path

from dmg.core.data.loaders import HydroLoader
from dmg.core.utils import print_config, set_randomseed
from dmg.core.utils.utils import initialize_config
from omegaconf import OmegaConf

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from implements import BaselineTrainer, build_causal_dpl
from train_causal_dpl import (
    _build_loader_config,
    _build_optimizer_and_scheduler,
    _log_effective_cluster_fold_sizes,
    _normalize_runtime_paths,
    _run_model_preflight,
    _validate_mc_mlp_config,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('train_kge_baseline')


def _resolve_path(path_str: str) -> str:
    path = Path(path_str)
    if path.exists():
        return str(path)

    repo_path = REPO_ROOT / path_str
    if repo_path.exists():
        return str(repo_path)

    project_path = PROJECT_DIR / path_str
    if project_path.exists():
        return str(project_path)

    return str(path)


def parse_args():
    p = argparse.ArgumentParser(description='KGE baseline training / evaluation')
    p.add_argument('--config', default='conf/config_kge_dhbv.yaml',
                   help='Path to dmg config yaml')
    p.add_argument('--holdout', type=str, default=None,
                   help='Effective cluster to hold out for OOD eval (A-G). '
                        'Overrides causal.holdout_cluster in config.')
    p.add_argument('--mode', default=None,
                   help='Override config mode: train | test | train_test')
    p.add_argument('--epochs', type=int, default=None,
                   help='Override train.epochs in config')
    p.add_argument('--seed', type=int, default=None,
                   help='Override experiment seed')
    p.add_argument('--mc-samples', type=int, default=None,
                   help='Override test.mc_samples in config')
    p.add_argument('--mc-selection-metric', default=None,
                   choices=('mse', 'rmse', 'loss', 'kge'),
                   help='Override test.mc_selection_metric in config')
    return p.parse_args()


def main():
    args = parse_args()

    raw_config = OmegaConf.load(_resolve_path(args.config))
    original_epochs = raw_config['train']['epochs']

    if args.mode:
        raw_config['mode'] = args.mode
    if args.holdout is not None:
        if raw_config.get('causal') is None:
            raw_config['causal'] = {}
        raw_config['causal']['holdout_cluster'] = args.holdout
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
    if args.mc_samples is not None:
        raw_config['test']['mc_samples'] = args.mc_samples
    if args.mc_selection_metric is not None:
        raw_config['test']['mc_selection_metric'] = args.mc_selection_metric

    _normalize_runtime_paths(raw_config)

    config = initialize_config(raw_config)
    _validate_mc_mlp_config(config)

    causal_cfg = config.get('causal', {})
    for required in ('cluster_csv', 'basin_ids_path'):
        if required not in causal_cfg:
            raise ValueError(
                f"Missing 'causal.{required}' in config. "
                "Baseline holdout evaluation requires the same causal split metadata "
                "used by train_causal_dpl.py."
            )

    holdout_cluster = causal_cfg.get('holdout_cluster')
    _log_effective_cluster_fold_sizes(causal_cfg)
    log.info(
        "Mode: %s  |  Held-out effective cluster: %s  |  Model dir: %s",
        config['mode'],
        holdout_cluster,
        config['model_dir'],
    )

    lr_sched = config['train'].get('lr_scheduler')
    if isinstance(lr_sched, dict):
        config['train']['lr_scheduler'] = lr_sched.get('name', str(lr_sched))
    print_config(config)
    if isinstance(lr_sched, dict):
        config['train']['lr_scheduler'] = lr_sched

    set_randomseed(config['seed'])

    log.info("Loading datasets...")
    data_loader = HydroLoader(
        _build_loader_config(config),
        test_split=True,
        overwrite=False,
    )

    log.info("Building CausalDplModel baseline...")
    model = build_causal_dpl(config)
    model = model.to(config['device'])
    log.info("Model device: %s", config['device'])
    _run_model_preflight(model, data_loader.train_dataset, config)
    optimizer, scheduler = _build_optimizer_and_scheduler(config, model)

    trainer = BaselineTrainer(
        config,
        model=model,
        train_dataset=data_loader.train_dataset,
        eval_dataset=data_loader.eval_dataset,
        optimizer=optimizer,
        scheduler=scheduler,
        write_out=True,
        verbose=False,
    )

    mode = config['mode']

    if 'train' in mode:
        log.info("Starting baseline ERM training...")
        trainer.train()
        log.info("Training stage finished. Model dir: %s", config['model_dir'])

    if 'test' in mode or mode == 'train_test':
        if holdout_cluster is not None:
            log.info("Evaluating on held-out effective cluster %s...", holdout_cluster)
            trainer.evaluate_holdout()
        else:
            log.info("Evaluating on full eval dataset (no holdout set)...")
            trainer.evaluate()
        log.info("Metrics saved to %s", config['output_dir'])


if __name__ == '__main__':
    main()
