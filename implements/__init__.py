"""Causal-dPL shared helpers outside the parameterize-local paper stack."""

from __future__ import annotations

from models.nns.fast_kan import FastKAN

_LAZY_EXPORTS = {
    'GnannEnvironmentSplitter': ('implements.gnann_splitter', 'GnannEnvironmentSplitter'),
    'CausalTrainer': ('implements.causal_trainer', 'CausalTrainer'),
    'BaselineTrainer': ('implements.baseline_trainer', 'BaselineTrainer'),
}


def build_causal_dpl(config: dict):
    """Build a DPL model from a config dict."""
    from dmg.models.neural_networks.ann import AnnModel
    from dmg.models.neural_networks.mlp import MlpModel
    from project.parameterize.implements.hbv_static import HbvStatic
    from project.parameterize.implements.mc_mlp import McMlpModel
    from project.parameterize.implements.my_dpl_model import MyDplModel

    phy_cfg = config['model']['phy']
    nn_cfg = config['model']['nn']

    phy_model = HbvStatic(config=phy_cfg, device=config['device'])
    nx = len(nn_cfg['forcings']) + len(nn_cfg['attributes'])
    ny = phy_model.learnable_param_count
    nn_name = str(nn_cfg.get('name', 'MlpModel'))

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
    elif nn_name in {'McMlpModel', 'McMlp', 'mc_mlp'}:
        if nn_cfg.get('forcings'):
            raise ValueError(
                "McMlpModel only supports static-basin inputs; set model.nn.forcings to []."
            )
        if (
            type(phy_model).__name__ == 'HbvStatic'
            and str(nn_cfg.get('output_activation', 'sigmoid')).lower() != 'sigmoid'
        ):
            raise ValueError(
                "HbvStatic expects normalized parameters in [0, 1]; "
                "use model.nn.output_activation='sigmoid' with McMlpModel."
            )
        nn_model = McMlpModel(
            nn_cfg,
            nx=nx,
            ny=ny,
        )
    elif nn_name in {'FastKAN', 'fast_kan', 'fastkan'}:
        if nn_cfg.get('forcings'):
            raise ValueError(
                "FastKAN only supports static-basin inputs; set model.nn.forcings to []."
            )
        if (
            type(phy_model).__name__ == 'HbvStatic'
            and str(nn_cfg.get('output_activation', 'sigmoid')).lower() != 'sigmoid'
        ):
            raise ValueError(
                "HbvStatic expects normalized parameters in [0, 1]; "
                "use model.nn.output_activation='sigmoid' with FastKAN."
            )
        nn_model = FastKAN(
            nn_cfg,
            nx=nx,
            ny=ny,
        )
    else:
        raise ValueError(f"Unsupported nn model '{nn_name}'.")

    return MyDplModel(
        phy_model=phy_model,
        nn_model=nn_model,
        config=config,
        device=config['device'],
    )


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = target
    module = __import__(module_name, fromlist=[attr_name])
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


__all__ = [
    'GnannEnvironmentSplitter',
    'CausalTrainer',
    'BaselineTrainer',
    'build_causal_dpl',
]
