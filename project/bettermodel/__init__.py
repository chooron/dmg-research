import logging
import os
from copy import deepcopy
from pathlib import Path
from typing import Any

import hydra
import torch
from omegaconf import OmegaConf

from dmg.core.utils import initialize_config

log = logging.getLogger(__name__)
PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parent.parent

__all__ = [
    'load_config',
    'take_data_sample',
]


def _normalize_none_like(value: Any) -> Any:
    if isinstance(value, str) and value.strip().lower() in {'', 'none', 'null'}:
        return None
    return value


def _first_present(mapping: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = mapping.get(key)
        if value is not None:
            return value
    return None


def _preserve_trailing_separator(original: str, resolved: Path) -> str:
    resolved_str = str(resolved)
    if original.endswith(("/", "\\")):
        return resolved_str.rstrip("/\\") + "/"
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


def _normalize_runtime_paths(config: dict[str, Any]) -> None:
    observations_cfg = config.get('observations')
    if observations_cfg:
        for key in ('data_path', 'gage_info', 'subset_path', 'train_path', 'test_path'):
            if observations_cfg.get(key):
                observations_cfg[key] = _resolve_input_path(observations_cfg[key])

    data_cfg = config.get('data')
    if data_cfg:
        for key in ('basin_ids_path', 'basin_ids_reference_path'):
            if data_cfg.get(key):
                data_cfg[key] = _resolve_input_path(data_cfg[key])


def _normalize_paths(config: dict[str, Any]) -> None:
    output_dir = config.get('output_dir') or config.get('save_path')
    trained_model = config.get('trained_model')
    if trained_model:
        trained_model = _resolve_input_path(trained_model, base_dir=PROJECT_DIR)
        config['trained_model'] = trained_model
    if (not output_dir) and trained_model:
        output_dir = os.path.dirname(os.path.normpath(trained_model))
    if not output_dir:
        output_dir = os.getcwd()
    else:
        output_dir = _resolve_output_path(output_dir)

    model_dir = config.get('model_dir') or trained_model or os.path.join(output_dir, 'model')
    plot_dir = config.get('plot_dir') or os.path.join(output_dir, 'plot')
    sim_dir = config.get('sim_dir') or config.get('out_path') or os.path.join(output_dir, 'sim')

    model_dir = _resolve_output_path(model_dir)
    plot_dir = _resolve_output_path(plot_dir)
    sim_dir = _resolve_output_path(sim_dir)

    config['output_dir'] = output_dir
    config['model_dir'] = model_dir
    config['plot_dir'] = plot_dir
    config['sim_dir'] = sim_dir

    # Legacy aliases used throughout bettermodel utilities.
    config['save_path'] = output_dir
    config['model_path'] = model_dir
    config['out_path'] = sim_dir


def _normalize_train_config(config: dict[str, Any]) -> None:
    train_cfg = config.setdefault('train', {})
    scheduler_name = _normalize_none_like(train_cfg.get('lr_scheduler'))
    scheduler_params = deepcopy(train_cfg.get('lr_scheduler_params') or {})
    loss_cfg = config.get('loss_function') or train_cfg.get('loss_function') or {}

    if isinstance(loss_cfg, str):
        loss_name = loss_cfg
        loss_cfg = {'name': loss_name, 'model': loss_name}
    else:
        loss_name = loss_cfg.get('name') or loss_cfg.get('model')
        loss_cfg = deepcopy(loss_cfg)
        if loss_name:
            loss_cfg.setdefault('name', loss_name)
            loss_cfg.setdefault('model', loss_name)

    lr_value = train_cfg.get('lr')
    if lr_value is None:
        lr_value = train_cfg.get('learning_rate')
    if lr_value is None:
        legacy_nn_cfg = (config.get('delta_model') or {}).get('nn_model', {})
        lr_value = legacy_nn_cfg.get('learning_rate')

    if lr_value is not None:
        train_cfg['lr'] = lr_value
        train_cfg['learning_rate'] = lr_value

    if loss_name:
        train_cfg['loss_function'] = {'name': loss_name, 'model': loss_name}
        config['loss_function'] = {'name': loss_name, 'model': loss_name}

    if scheduler_name is None:
        train_cfg['lr_scheduler'] = None
    else:
        train_cfg['lr_scheduler'] = {
            'name': scheduler_name,
            **scheduler_params,
        }
    train_cfg['lr_scheduler_params'] = scheduler_params


def _normalize_model_config(config: dict[str, Any]) -> None:
    delta_model = config.get('delta_model')
    if delta_model is None and config.get('model') is None:
        raise KeyError("Configuration must define 'delta_model' or 'model'.")

    if config.get('model') is None and delta_model is not None:
        phy_cfg = deepcopy(delta_model.get('phy_model') or {})
        nn_cfg = deepcopy(delta_model.get('nn_model') or {})

        phy_names = deepcopy(phy_cfg.get('name') or phy_cfg.get('model') or [])
        phy_cfg.setdefault('name', phy_names)
        phy_cfg.setdefault('model', deepcopy(phy_names))

        dynamic_params = phy_cfg.get('dynamic_params')
        if isinstance(dynamic_params, list):
            phy_cfg['dynamic_params'] = {phy_names[0]: dynamic_params} if phy_names else {}

        nn_name = nn_cfg.get('name') or nn_cfg.get('model')
        if nn_name is not None:
            nn_cfg['name'] = nn_name

        if nn_cfg.get('dropout') is None:
            dropout = _first_present(
                nn_cfg,
                ['dropout', 'mlp_dropout', 'lstm_dropout', 'gru_dropout', 'tcn_dropout'],
            )
            if dropout is not None:
                nn_cfg['dropout'] = dropout

        if nn_cfg.get('hidden_size') is None:
            hidden_size = _first_present(
                nn_cfg,
                [
                    'hidden_size',
                    'mlp_hidden_size',
                    'lstm_hidden_size',
                    'gru_hidden_size',
                    'tcn_hidden_size',
                    'transformer_d_model',
                ],
            )
            if hidden_size is not None:
                nn_cfg['hidden_size'] = hidden_size

        config['model'] = {
            'rho': delta_model['rho'],
            'warm_up': phy_cfg.get('warm_up', delta_model['rho']),
            'use_log_norm': deepcopy(
                phy_cfg.get('use_log_norm', delta_model.get('use_log_norm', [])),
            ),
            'phy': phy_cfg,
            'nn': nn_cfg,
        }

    model_cfg = config['model']
    if model_cfg.get('warm_up') is None:
        model_cfg['warm_up'] = model_cfg.get('warmup', delta_model['rho'] if delta_model else 0)
    model_cfg['warmup'] = model_cfg.get('warm_up', 0)

    phy_cfg = deepcopy(model_cfg.get('phy') or {})
    nn_cfg = deepcopy(model_cfg.get('nn') or {})
    phy_names = deepcopy(phy_cfg.get('name') or phy_cfg.get('model') or [])
    phy_cfg.setdefault('name', phy_names)
    phy_cfg.setdefault('model', deepcopy(phy_names))

    dynamic_params = phy_cfg.get('dynamic_params')
    if isinstance(dynamic_params, list):
        phy_cfg['dynamic_params'] = {phy_names[0]: dynamic_params} if phy_names else {}

    nn_name = nn_cfg.get('name') or nn_cfg.get('model')
    if nn_name is not None:
        nn_cfg['name'] = nn_name

    scheduler_cfg = config['train'].get('lr_scheduler')
    scheduler_name = (
        scheduler_cfg.get('name') if isinstance(scheduler_cfg, dict) else _normalize_none_like(scheduler_cfg)
    )
    scheduler_params = deepcopy(config['train'].get('lr_scheduler_params') or {})

    config['model']['phy'] = phy_cfg
    config['model']['nn'] = nn_cfg
    config['delta_model'] = {
        'rho': model_cfg['rho'],
        'phy_model': deepcopy(phy_cfg),
        'nn_model': {
            **deepcopy(nn_cfg),
            'model': nn_cfg.get('name', nn_cfg.get('model')),
            'learning_rate': config['train'].get('learning_rate'),
            'lr_scheduler': scheduler_name,
            'lr_scheduler_params': scheduler_params,
        },
        'train': {
            'lr_scheduler': scheduler_name,
            'lr_scheduler_params': scheduler_params,
        },
    }


def _normalize_bettermodel_config(config: dict[str, Any]) -> dict[str, Any]:
    config = deepcopy(config)

    if config.get('mode') == 'simulation':
        config['mode'] = 'sim'

    if 'sim' not in config and 'simulation' in config:
        config['sim'] = deepcopy(config['simulation'])
    if 'simulation' not in config and 'sim' in config:
        config['simulation'] = deepcopy(config['sim'])

    seed = config.get('seed', config.get('random_seed', 0))
    config['seed'] = seed
    config['random_seed'] = seed

    config.setdefault('name', 'bettermodel')
    config.setdefault('multimodel_type', None)
    config.setdefault('logging', None)
    config.setdefault('cache_states', False)
    config.setdefault('verbose', False)

    _normalize_runtime_paths(config)
    _normalize_train_config(config)
    _normalize_model_config(config)
    _normalize_paths(config)

    return config


def load_config(path: str) -> dict[str, Any]:
    """Parse and initialize configuration settings from yaml with Hydra.

    This loader is capable of handling config files in nonlinear directory
    structures.

    Parameters
    ----------
    config_path
        Path to the configuration file.

    Returns
    -------
    dict
        Formatted configuration settings.
    """
    path_obj = Path(path).resolve()
    path_no_ext = path_obj.with_suffix('')
    config_name = path_no_ext.name
    config_dir = str(path_no_ext.parent)

    with hydra.initialize_config_dir(config_dir=config_dir, version_base='1.3'):
        config = hydra.compose(config_name=config_name)

    # Convert the OmegaConf object to a dict.
    config = OmegaConf.to_container(config, resolve=True)
    config = _normalize_bettermodel_config(config)

    # Convert date ranges / set device and dtype / create output dirs.
    config = initialize_config(config)
    _normalize_paths(config)
    config['random_seed'] = config['seed']
    config['simulation'] = deepcopy(config['sim'])

    return config


def take_data_sample(
        config: dict,
        dataset_dict: dict[str, torch.Tensor],
        days: int = 730,
        basins: int = 100,
) -> dict[str, torch.Tensor]:
    """Take sample of data.

    Parameters
    ----------
    config
        Configuration settings.
    dataset_dict
        Dictionary containing dataset tensors.
    days
        Number of days to sample.
    basins
        Number of basins to sample.

    Returns
    -------
    dict
        Dictionary containing sampled dataset tensors.
    """
    dataset_sample = {}

    for key, value in dataset_dict.items():
        if value.ndim == 3:
            # Determine warm-up period based on the key
            if key in ['x_phy', 'xc_nn_norm']:
                warm_up = 0
            else:
                warm_up = config['delta_model']['phy_model']['warm_up']

            # Clone and detach the tensor to avoid the warning
            dataset_sample[key] = value[warm_up:days, :basins, :].clone().detach().to(
                dtype=torch.float32, device=config['device'])

        elif value.ndim == 2:
            # Clone and detach the tensor to avoid the warning
            dataset_sample[key] = value[:basins, :].clone().detach().to(
                dtype=torch.float32, device=config['device'])

        else:
            raise ValueError(f"Incorrect input dimensions. {key} array must have 2 or 3 dimensions.")

    # Adjust the 'target' tensor based on the configuration
    if ('HBV1_1p' in config['delta_model']['phy_model']['model'] and
            config['delta_model']['phy_model']['use_warmup_mode'] and
            config['multimodel_type'] == 'none'):
        pass  # Keep 'warmup' days for dHBV1.1p
    else:
        warm_up = config['delta_model']['phy_model']['warm_up']
        dataset_sample['target'] = dataset_sample['target'][warm_up:days, :basins]

    return dataset_sample
