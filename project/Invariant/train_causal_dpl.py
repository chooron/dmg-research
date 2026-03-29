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
import copy
import logging
import os
import sys
from pathlib import Path

import torch

from dmg.core.data.loaders import HydroLoader
from dmg.core.utils import set_randomseed, print_config
from omegaconf import OmegaConf
from dmg.core.utils.utils import initialize_config

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(REPO_ROOT))

from implements import build_causal_dpl, CausalTrainer
from implements.gnann_splitter import GnannEnvironmentSplitter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s  %(levelname)-8s  %(name)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('train_causal_dpl')


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


def _preserve_trailing_separator(original: str, resolved: Path) -> str:
    resolved_str = str(resolved)
    if original.endswith(('/', '\\')):
        return resolved_str.rstrip('/\\') + '/'
    return resolved_str


def _resolve_input_path(path_str: str, base_dir: Path = REPO_ROOT) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return _preserve_trailing_separator(path_str, path)

    candidates = [
        Path.cwd() / path,
        base_dir / path,
        PROJECT_DIR / path,
    ]
    for candidate in candidates:
        if candidate.exists():
            return _preserve_trailing_separator(path_str, candidate.resolve())

    return _preserve_trailing_separator(path_str, (base_dir / path).resolve())


def _resolve_output_path(path_str: str, base_dir: Path = PROJECT_DIR) -> str:
    path = Path(path_str).expanduser()
    if path.is_absolute():
        return _preserve_trailing_separator(path_str, path)
    return _preserve_trailing_separator(path_str, (base_dir / path).resolve())


def _normalize_runtime_paths(raw_config) -> None:
    observations_cfg = raw_config.get('observations')
    if observations_cfg and observations_cfg.get('data_path'):
        observations_cfg['data_path'] = _resolve_input_path(observations_cfg['data_path'])

    causal_cfg = raw_config.get('causal')
    if causal_cfg:
        for key in ('cluster_csv', 'basin_ids_path'):
            if causal_cfg.get(key):
                causal_cfg[key] = _resolve_input_path(causal_cfg[key])

    if raw_config.get('output_dir'):
        raw_config['output_dir'] = _resolve_output_path(raw_config['output_dir'])


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
    p.add_argument('--mc-samples', type=int, default=None,
                   help='Override test.mc_samples in config')
    p.add_argument('--mc-selection-metric', default=None,
                   choices=('mse', 'rmse', 'loss', 'kge'),
                   help='Override test.mc_selection_metric in config')
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


def _build_loader_config(config: dict) -> dict:
    """Keep full datasets on CPU; samplers move only active batches to GPU."""
    loader_config = copy.deepcopy(config)
    loader_config['device'] = 'cpu'
    return loader_config


def _validate_mc_mlp_config(config: dict) -> None:
    model_cfg = config['model']
    phy_cfg = model_cfg.get('phy', {})
    nn_cfg = model_cfg.get('nn', {})
    nn_name = str(nn_cfg.get('name', ''))
    if nn_name not in {'McMlpModel', 'McMlp', 'mc_mlp'}:
        return

    if nn_cfg.get('forcings'):
        raise ValueError(
            "McMlpModel is configured for static basin attributes only; "
            "set model.nn.forcings to []."
        )

    output_activation = str(nn_cfg.get('output_activation', 'sigmoid')).lower()
    if output_activation != 'sigmoid':
        raise ValueError(
            "McMlpModel must use output_activation='sigmoid' so HbvStatic can map "
            "normalized parameters into physical ranges."
        )

    if int(phy_cfg.get('nmul', 1) or 1) != 1:
        raise ValueError(
            "This static-attribute McMlpModel setup expects model.phy.nmul == 1."
        )

    if not nn_cfg.get('attributes'):
        raise ValueError(
            "McMlpModel requires at least one static attribute in model.nn.attributes."
        )


def _run_model_preflight(
    model: torch.nn.Module,
    dataset: dict[str, torch.Tensor],
    config: dict,
) -> None:
    if dataset is None:
        raise ValueError("Training dataset is required for model preflight.")

    if 'xc_nn_norm' not in dataset or 'x_phy' not in dataset:
        raise KeyError("Preflight expects dataset keys 'xc_nn_norm' and 'x_phy'.")

    xc_nn_norm = dataset['xc_nn_norm']
    x_phy = dataset['x_phy']
    if xc_nn_norm.ndim != 3:
        raise ValueError(
            f"xc_nn_norm must have shape [T, B, nx], got {tuple(xc_nn_norm.shape)}."
        )
    if x_phy.ndim != 3:
        raise ValueError(
            f"x_phy must have shape [T, B, n_forcings], got {tuple(x_phy.shape)}."
        )

    expected_nx = (
        len(config['model']['nn'].get('forcings', []))
        + len(config['model']['nn'].get('attributes', []))
    )
    if xc_nn_norm.shape[-1] != expected_nx:
        raise ValueError(
            "Dataset/config feature mismatch for xc_nn_norm: "
            f"{xc_nn_norm.shape[-1]} vs expected {expected_nx}."
        )

    expected_forcings = len(config['model']['phy'].get('forcings', []))
    if x_phy.shape[-1] != expected_forcings:
        raise ValueError(
            "Dataset/config forcing mismatch for x_phy: "
            f"{x_phy.shape[-1]} vs expected {expected_forcings}."
        )

    sample_basin_count = min(2, xc_nn_norm.shape[1])
    sample = {
        'xc_nn_norm': xc_nn_norm[:, :sample_basin_count].to(config['device']),
        'x_phy': x_phy[:, :sample_basin_count].to(config['device']),
    }

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            parameters = model.nn_model(sample['xc_nn_norm'])
            expected_ny = model.phy_model.learnable_param_count

            if parameters.ndim != 3:
                raise ValueError(
                    "NN output must have shape [T, B, ny] for the CausalDplModel/HbvStatic "
                    f"interface, got {tuple(parameters.shape)}."
                )
            if parameters.shape[:2] != sample['xc_nn_norm'].shape[:2]:
                raise ValueError(
                    "NN output time/basin dimensions do not match xc_nn_norm: "
                    f"{tuple(parameters.shape[:2])} vs {tuple(sample['xc_nn_norm'].shape[:2])}."
                )
            if parameters.shape[-1] != expected_ny:
                raise ValueError(
                    "NN output size does not match phy_model.learnable_param_count: "
                    f"{parameters.shape[-1]} vs {expected_ny}."
                )

            if type(model.nn_model).__name__ == 'McMlpModel':
                if not parameters.is_contiguous():
                    raise ValueError("McMlpModel output must be contiguous.")
                if not torch.allclose(parameters[0], parameters[-1]):
                    raise ValueError(
                        "McMlpModel output is expected to be basin-static and repeated across time."
                    )

            predictions = model(sample, eval=True)
            if 'streamflow' not in predictions:
                raise KeyError("Model prediction dictionary must contain 'streamflow'.")

            streamflow = predictions['streamflow']
            if streamflow.ndim != 3:
                raise ValueError(
                    f"streamflow must have shape [T_pred, B, 1], got {tuple(streamflow.shape)}."
                )
            if streamflow.shape[1] != sample_basin_count or streamflow.shape[2] != 1:
                raise ValueError(
                    "Unexpected streamflow output shape: "
                    f"{tuple(streamflow.shape)} for basin count {sample_basin_count}."
                )

        log.info(
            "Model preflight passed: xc_nn_norm %s -> parameters %s -> streamflow %s",
            tuple(sample['xc_nn_norm'].shape),
            tuple(parameters.shape),
            tuple(streamflow.shape),
        )
    finally:
        model.train(was_training)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # 1. Load config and apply CLI overrides before interpolation resolves.
    raw_config = OmegaConf.load(_resolve_path(args.config))
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
    if args.mc_samples is not None:
        raw_config['test']['mc_samples'] = args.mc_samples
    if args.mc_selection_metric is not None:
        raw_config['test']['mc_selection_metric'] = args.mc_selection_metric

    _normalize_runtime_paths(raw_config)

    config = initialize_config(raw_config)
    _validate_mc_mlp_config(config)

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
    data_loader = HydroLoader(
        _build_loader_config(config),
        test_split=True,
        overwrite=False,
    )

    # 3. Build model
    log.info("Building CausalDplModel...")
    model = build_causal_dpl(config)
    model = model.to(config['device'])
    log.info(f"Model device: {config['device']}")
    _run_model_preflight(model, data_loader.train_dataset, config)
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
        log.info(f"Training stage finished. Model dir: {config['model_dir']}")

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
