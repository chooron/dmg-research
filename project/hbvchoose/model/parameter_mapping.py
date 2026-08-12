"""Normalized-parameter to physical-value mapping compatible with HbvStatic.

Replicates the logic of ``hydrodl2.core.calc.change_param_range`` using
the same parameter_bounds as HbvStatic.
"""

import torch

from model.hbv_static import HbvStatic


def change_param_range(value: torch.Tensor, bounds: list[float]) -> torch.Tensor:
    """Map a value in [0, 1] to the physical range [bounds[0], bounds[1]].

    Identical behaviour to ``hydrodl2.core.calc.change_param_range``.
    """
    lo, hi = bounds
    return lo + (hi - lo) * value


class ParameterMapper:
    """Maps normalized parameters [0,1] to physical parameter dicts.

    Compatible with HbvFormulaStatic's formula_config parameter expectations.
    """

    # Maps HbvStatic par-prefix names to formula-pool parameter names
    PARAM_ALIAS = {
        "parBETA": "beta",
        "parFC": "FC",
        "parK0": "K_0",
        "parK1": "K_1",
        "parK2": "K_2",
        "parLP": "LP",
        "parPERC": "PERC",
        "parUZL": "UZL",
        "parTT": "TT",
        "parCFMAX": "CFMAX",
        "parCFR": "CFR",
        "parCWH": "CWH",
    }

    # Node-specific parameter assignments
    NODE_PARAMS = {
        "snow": ["parTT", "parCFMAX", "parCFR", "parCWH"],
        "recharge": ["parFC", "parBETA"],
        "aet": ["parFC", "parLP"],
        "response": ["parK0", "parK1", "parK2", "parUZL", "parPERC"],
    }

    def __init__(self, nmul: int = 1):
        self._hbv = HbvStatic(config={"nmul": nmul})
        self.nmul = nmul
        self.N_PHY = len(self._hbv.parameter_bounds)
        self.N_ROUTE = len(self._hbv.routing_parameter_bounds)
        self.PARAMETER_NAMES = tuple(self._hbv.parameter_bounds)
        self.ROUTING_PARAMETER_NAMES = tuple(self._hbv.routing_parameter_bounds)

    def normalized_to_physical(
        self, normalized: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        """Convert normalized params to physical dicts.

        Args:
            normalized: tensor shape (basin,) or (1, basin, N_PHY + N_ROUTE)

        Returns:
            phy: {parBETA: tensor, ...}  — physical HbvStatic parameter values
            route: {route_a: tensor, route_b: tensor}
        """
        if normalized.ndim == 3:
            pvec = normalized[-1]  # (basin, P)
        else:
            pvec = normalized  # (basin, P)

        basin_count = pvec.shape[0]
        p = pvec[:, : self.N_PHY * self.nmul].view(basin_count, self.N_PHY, self.nmul)

        phy = {}
        for idx, (name, bounds) in enumerate(self._hbv.parameter_bounds.items()):
            phy[name] = change_param_range(p[:, idx, :], bounds)

        r = pvec[:, self.N_PHY * self.nmul :]
        route = {}
        for idx, (name, bounds) in enumerate(self._hbv.routing_parameter_bounds.items()):
            route[name] = change_param_range(r[:, idx], bounds)

        return phy, route

    def physical_to_formula_params(
        self,
        formula_config: dict[str, str],
        phy: dict[str, torch.Tensor],
    ) -> dict[str, dict[str, torch.Tensor]]:
        """Map HbvStatic physical params to node-specific formula-pool param dicts.

        Returns {node: {param_name: tensor_value}} suitable for HbvFormulaStatic.
        """
        result = {}
        for node in formula_config:
            node_params = {}
            for hbv_name in self.NODE_PARAMS.get(node, []):
                if hbv_name in phy:
                    alias = self.PARAM_ALIAS[hbv_name]
                    node_params[alias] = phy[hbv_name].squeeze(-1) if phy[hbv_name].ndim > 1 else phy[hbv_name]
            result[node] = node_params
        if "parPERC" in phy:
            result["_perc"] = phy["parPERC"].squeeze(-1)
        return result
