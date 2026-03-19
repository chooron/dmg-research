"""
train_causal_dpl.py
-------------------
Main entry point for Causal-dPL group-holdout cross-validation.

Usage
-----
    python train_causal_dpl.py --config conf/config_dhbv.yaml \
                                --holdout 1          # hold out Group 1
                                --mode train_test    # train + OOD eval

    # Run all 4 group holdouts sequentially:
    for g in 1 2 3 4; do
        python train_causal_dpl.py --config conf/config_dhbv.yaml --holdout $g
    done

Config extensions (add to your yaml under a 'causal:' key)
-----------------------------------------------------------
causal:
  cluster_csv:    data/gauge_pos_cluster_climate.csv
  basin_ids_path: data/gage_id.txt          # or gage_id.npy
  use_groups:     true                      # 4 coarse groups (vs 10 clusters)
  holdout_group:  1                         # overridden by --holdout CLI arg

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
    p.add_argument('--holdout',  type=int, default=None,
                   help='Group to hold out for OOD eval (1-4). '
                        'Overrides causal.holdout_group in config.')
    p.add_argument('--mode',     default=None,
                   help='Override config mode: train | test | train_test')
    p.add_argument('--lambda-irm', type=float, default=None,
                   help='Override lambda_irm in config')
    p.add_argument('--epochs',   type=int, default=None,
                   help='Override train.epochs in config')
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Load config
    config = initialize_config(OmegaConf.load(args.config))

    # Apply CLI overrides
    if args.mode:
        config['mode'] = args.mode
    if args.holdout is not None:
        config.setdefault('causal', {})['holdout_group'] = args.holdout
    if args.lambda_irm is not None:
        config['lambda_irm'] = args.lambda_irm
    if args.epochs is not None:
        config['train']['epochs'] = args.epochs

    # Validate causal config block
    causal_cfg = config.get('causal', {})
    for required in ('cluster_csv', 'basin_ids_path'):
        if required not in causal_cfg:
            raise ValueError(
                f"Missing 'causal.{required}' in config. "
                "Add a 'causal:' block to your yaml (see module docstring)."
            )

    holdout_group = causal_cfg.get('holdout_group')
    log.info(f"Mode: {config['mode']}  |  Holdout group: {holdout_group}")
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

    # 4. Build trainer
    trainer = CausalTrainer(
        config,
        model=model,
        train_dataset=data_loader.train_dataset,
        eval_dataset=data_loader.eval_dataset,
        write_out=True,
        verbose=False,
    )

    # 5. Run
    mode = config['mode']

    if 'train' in mode:
        log.info("Starting IRM training...")
        trainer.train()
        log.info(f"Training complete. Model saved to {config['model_dir']}")

    if 'test' in mode or mode == 'train_test':
        if holdout_group is not None:
            log.info(f"Evaluating on held-out Group {holdout_group}...")
            trainer.evaluate_holdout()
        else:
            log.info("Evaluating on full eval dataset (no holdout set)...")
            trainer.evaluate()
        log.info(f"Metrics saved to {config['output_dir']}")


if __name__ == '__main__':
    main()
