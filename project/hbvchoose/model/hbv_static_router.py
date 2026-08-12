"""HbvStaticFormulaRouter — StaticFormulaRouter + HbvFormulaStatic integration.

Wraps a StaticFormulaRouter with an HbvFormulaStatic backend.  Per-basin
formula selections drive the HBV simulation.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model.formula_pool import CandidateFormulaPool
from model.hbv_formula_static import HbvFormulaStatic
from model.parameter_mapping import ParameterMapper
from model.static_formula_router import StaticFormulaRouter

_NODE_ORDER = ["snow", "recharge", "aet", "response"]
_EXTRA_PARAMS = {
    "S4": {"a_s": 0.3, "phi_s": 172.0},
    "S5": {"c_m": 0.3},
    "R4": {"a_r": 10.0, "c_r": 0.5},
    "R5": {"b_v": 1.0},
    "E3": {"gamma_E": 1.2},
    "E4": {"s_w": 0.1, "s_o": 0.6},
    "Q2": {"alpha_Q": 1.2},
}


class HbvStaticFormulaRouter(nn.Module):
    """HBV model driven by a static formula router.

    Parameters
    ----------
    attr_dim:
        Number of static basin-attribute features.
    hidden_dim:
        Reserved (passed through to ``StaticFormulaRouter``).
    temperature:
        Logit temperature for ``StaticFormulaRouter``.
    default_bias:
        Default-HBV anchor bias for ``StaticFormulaRouter``.
    hard_eval:
        Use hard argmax during evaluation.
    warm_up:
        Spin-up steps for ``HbvFormulaStatic.simulate``.
    nearzero:
        Minimum storage value to avoid vanishing gradients.
    """

    def __init__(
        self,
        attr_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        default_bias: float = 2.0,
        hard_eval: bool = True,
        warm_up: int = 20,
        nearzero: float = 1e-5,
    ) -> None:
        super().__init__()
        self.router = StaticFormulaRouter(
            attr_dim=attr_dim,
            hidden_dim=hidden_dim,
            temperature=temperature,
            default_bias=default_bias,
            hard_eval=hard_eval,
        )
        self.warm_up = warm_up
        self.nearzero = nearzero
        self._pool = CandidateFormulaPool()
        self._mapper = ParameterMapper(nmul=1)

    def forward(
        self,
        forcing: torch.Tensor,
        attrs: torch.Tensor,
        normalized_params: torch.Tensor | None = None,
    ) -> dict:
        """Run the HBV model with per-basin formula selections.

        Parameters
        ----------
        forcing: ``[T, B, F]`` float tensor of meteorological forcings.
            Channels are (prcp, tmean, pet).
        attrs: ``[B, attr_dim]`` float tensor of static basin attributes.
        normalized_params: ``[B, 16]`` float tensor of normalized HBV
            parameters in [0,1].  If ``None``, uses default fixed parameters.

        Returns
        -------
        dict with ``Qsim``, ``Q_raw``, ``router``, ``diagnostics``,
        ``water_balance``.
        """
        device = forcing.device
        B = attrs.shape[0]

        router_out = self.router(attrs)

        fids_dict = {n: self._pool.formulas(n, "main") for n in _NODE_ORDER}

        selected_ids: dict[str, list[str]] = {}
        for node in _NODE_ORDER:
            idx = router_out["selected"][node]
            fids = fids_dict[node]
            selected_ids[node] = [fids[int(i.item())] for i in idx]

        has_params = normalized_params is not None
        if has_params:
            if normalized_params.ndim == 1:
                normalized_params = normalized_params.unsqueeze(0)
            phy_all, _ = self._mapper.normalized_to_physical(normalized_params)

        Qsim_list: list[torch.Tensor] = []
        Q_raw_list: list[torch.Tensor] = []
        diag_list: list[dict] = []

        for b in range(B):
            combo = {n: selected_ids[n][b] for n in _NODE_ORDER}
            Pb = forcing[:, b, 0]
            Tb = forcing[:, b, 1]
            PETb = forcing[:, b, 2]

            params: dict = {}
            if has_params:
                for n in _NODE_ORDER:
                    node_params = {}
                    for hbv_name in self._mapper.NODE_PARAMS.get(n, []):
                        if hbv_name in phy_all:
                            alias = self._mapper.PARAM_ALIAS[hbv_name]
                            val = phy_all[hbv_name]
                            val_b = val[b]
                            if val_b.ndim >= 1:
                                val_b = val_b.squeeze()
                            node_params[alias] = val_b
                    params[n] = node_params
                    fid = combo[n]
                    if fid in _EXTRA_PARAMS:
                        params[n].update(_EXTRA_PARAMS[fid])
                if "parPERC" in phy_all:
                    val = phy_all["parPERC"][b]
                    params["_perc"] = val

            model = HbvFormulaStatic(
                formula_config=combo,
                warm_up=self.warm_up,
                nearzero=self.nearzero,
                param_dicts=params,
            )
            diag = model.simulate(Pb, Tb, PETb)
            qs = diag["Qsim"] if diag.get("routing_applied", False) else diag["Q_raw"]
            Qsim_list.append(qs)
            Q_raw_list.append(diag["Q_raw"])
            diag_list.append(diag)

        max_len = max(q.shape[0] for q in Qsim_list)
        Qsim = torch.zeros(max_len, B, device=device)
        Q_raw = torch.zeros(max_len, B, device=device)
        for b in range(B):
            L = Qsim_list[b].shape[0]
            Qsim[:L, b] = Qsim_list[b]
            Q_raw[:L, b] = Q_raw_list[b]

        wb_residual = [
            d.get("water_balance_residual", 0.0) for d in diag_list
        ]
        wb_rel = [
            d.get("relative_water_balance_error", 0.0) for d in diag_list
        ]

        return {
            "Qsim": Qsim,
            "Q_raw": Q_raw,
            "router": router_out,
            "diagnostics": diag_list,
            "water_balance": {
                "residual": wb_residual,
                "relative_error": wb_rel,
            },
        }

    @property
    def formula_ids(self) -> dict[str, list[str]]:
        return self.router.formula_ids
