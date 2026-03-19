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
    phy_cfg = config['model']['phy']
    nn_cfg = config['model']['nn']
    phy_model = HbvStatic(config=phy_cfg, device=config['device'])
    nx = len(nn_cfg['forcings']) + len(nn_cfg['attributes'])
    ny = phy_model.learnable_param_count
    nn_model = AnnModel(
        nx=nx,
        ny=ny,
        hidden_size=nn_cfg['hidden_size'],
        dr=nn_cfg.get('dropout', 0.5),
    )
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
