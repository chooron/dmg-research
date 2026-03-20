"""Causal-dPL: IRM-augmented differentiable parameter learning for PUB.

Public API re-exported from submodules for backwards compatibility.
"""

from implements.hbv_static import HbvStatic
from implements.gnann_splitter import GnannEnvironmentSplitter
from implements.causal_dpl_model import CausalDplModel
from implements.causal_trainer import CausalTrainer


def build_causal_dpl(config: dict) -> CausalDplModel:
    """Build a CausalDplModel from a config dict."""
    from dmg.models.neural_networks.ann import AnnModel
    from dmg.models.neural_networks.mlp import MlpModel

    phy_cfg = config['model']['phy']
    nn_cfg = config['model']['nn']
    phy_model = HbvStatic(config=phy_cfg, device=config['device'])
    nx = len(nn_cfg['forcings']) + len(nn_cfg['attributes'])
    ny = phy_model.learnable_param_count
    nn_name = nn_cfg.get('name', 'MlpModel')

    if nn_name == 'AnnModel':
        nn_model = AnnModel(
            nx=nx,
            ny=ny,
            hidden_size=nn_cfg['hidden_size'],
            dr=nn_cfg.get('dropout', 0.5),
        )
    elif nn_name == 'MlpModel':
        nn_model = MlpModel(
            {'hidden_size': nn_cfg['hidden_size']},
            nx=nx,
            ny=ny,
        )
    else:
        raise ValueError(f"Unsupported nn model '{nn_name}'.")

    return CausalDplModel(
        phy_model=phy_model,
        nn_model=nn_model,
        config=config,
        device=config['device'],
    )


__all__ = [
    'HbvStatic',
    'GnannEnvironmentSplitter',
    'CausalDplModel',
    'CausalTrainer',
    'build_causal_dpl',
]
