"""
train_kge_baseline.py
---------------------
Baseline training with KgeBatchLoss (pure ERM, no IRM/VREx penalty).
Used to benchmark maximum throughput of HbvStatic + AnnModel.

Usage
-----
    python train_kge_baseline.py --config conf/config_kge_dhbv.yaml
    python train_kge_baseline.py --config conf/config_kge_dhbv.yaml --epochs 10
"""

import argparse
import logging
import os
import sys
import time

import torch
import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from omegaconf import OmegaConf
from dmg.core.utils.utils import initialize_config
from dmg.core.utils import set_randomseed
from dmg.core.data.loaders import HydroLoader
from dmg.core.data.data import create_training_grid
from dmg.core.data.samplers import HydroSampler
from dmg.models.criterion.kge_batch_loss import KgeBatchLoss

from implements import build_causal_dpl

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('train_kge_baseline')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config',  default='conf/config_kge_dhbv.yaml')
    p.add_argument('--epochs',  type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    config = initialize_config(OmegaConf.load(args.config))

    lr_sched = config['train'].get('lr_scheduler')
    if isinstance(lr_sched, dict):
        config['train']['lr_scheduler'] = lr_sched

    if args.epochs is not None:
        config['train']['epochs'] = args.epochs

    set_randomseed(config['seed'])

    log.info("Loading data...")
    data_loader = HydroLoader(config, test_split=False, overwrite=False)
    train_dataset = data_loader.train_dataset

    log.info("Building model...")
    model = build_causal_dpl(config).to(config['device'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['train']['lr'])
    lr_cfg = config['train'].get('lr_scheduler', {})
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=lr_cfg.get('T_max', config['train']['epochs']),
        eta_min=lr_cfg.get('eta_min', 1e-5),
    )
    loss_func = KgeBatchLoss(config, device=config['device'])
    sampler   = HydroSampler(config)

    n_samples, n_minibatch, n_timesteps = create_training_grid(
        train_dataset['xc_nn_norm'], config,
    )
    epochs = config['train']['epochs']
    log.info(f"Training: {epochs} epochs, {n_minibatch} minibatches/epoch, "
             f"{n_samples} samples, rho={n_timesteps}")

    total_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        total_loss  = 0.0
        model.train()

        prog_bar = tqdm.tqdm(
            range(1, n_minibatch + 1),
            desc=f"Epoch {epoch}/{epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        for _ in prog_bar:
            sample = sampler.get_training_sample(train_dataset, n_samples, n_timesteps)
            pred   = model(sample)
            y_pred = pred['streamflow'].squeeze(-1)
            warm_up = getattr(model.phy_model, 'warm_up', 0)
            y_obs  = sample['target'][-y_pred.shape[0]:, :, 0]

            loss = loss_func(y_pred=y_pred, y_obs=y_obs)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        scheduler.step()
        elapsed = time.perf_counter() - epoch_start
        avg_loss = total_loss / n_minibatch
        log.info(f"Epoch {epoch:3d}/{epochs}  loss={avg_loss:.6f}  "
                 f"time={elapsed:.1f}s  ({elapsed/n_minibatch:.2f}s/batch)")

    total_elapsed = time.perf_counter() - total_start
    log.info(f"Done. Total time: {total_elapsed:.1f}s  "
             f"({total_elapsed/epochs:.1f}s/epoch)")


if __name__ == '__main__':
    main()
