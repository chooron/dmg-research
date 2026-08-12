"""StaticFormulaRouter — node-wise formula selection from static basin attributes.

Reads formula candidates automatically from FORMULA_REGISTRY.  Each process
node (snow / recharge / aet / response) gets its own projection head.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.formula_pool import CandidateFormulaPool

_DEFAULT_ANCHOR = {"snow": "S0", "recharge": "R0", "aet": "E0", "response": "Q0"}
_NODE_ORDER = ["snow", "recharge", "aet", "response"]


class StaticFormulaRouter(nn.Module):
    """Node-wise static formula router.

    Each process node has an independent linear head that maps static basin
    attributes to per-formula logits.  ``recharge`` always uses hard (one-hot)
    routing with straight-through estimator during training.  ``snow`` / ``aet``
    / ``response`` use soft weights during training and hard argmax during
    evaluation (when *hard_eval* is ``True``).

    Parameters
    ----------
    attr_dim:
        Number of static basin-attribute features.
    hidden_dim:
        Unused in current linear-only version (reserved for future MLP heads).
    temperature:
        Temperature divisor applied to logits before softmax.
    default_bias:
        Bias added to the entry of each head that corresponds to the default
        HBV formula (S0 / R0 / E0 / Q0).
    hard_eval:
        If ``True`` (default), ``snow``/``aet``/``response`` use hard argmax
        during evaluation.  Soft weights are kept otherwise.
    """

    def __init__(
        self,
        attr_dim: int,
        hidden_dim: int = 64,
        temperature: float = 1.0,
        default_bias: float = 2.0,
        hard_eval: bool = True,
    ) -> None:
        super().__init__()
        _ = hidden_dim  # reserved

        self.attr_dim = attr_dim
        self.temperature = temperature
        self.default_bias = default_bias
        self.hard_eval = hard_eval

        self._pool = CandidateFormulaPool()
        self._formula_ids: dict[str, list[str]] = {}
        self._num_formulas: dict[str, int] = {}
        self.heads = nn.ModuleDict()

        for node in _NODE_ORDER:
            fids = self._pool.formulas(node, "main")
            self._formula_ids[node] = list(fids)
            self._num_formulas[node] = len(fids)

            default_fid = _DEFAULT_ANCHOR[node]
            default_idx = fids.index(default_fid)

            head = nn.Linear(attr_dim, len(fids))
            nn.init.xavier_uniform_(head.weight, gain=0.5)
            nn.init.zeros_(head.bias)
            head.bias.data[default_idx] = default_bias
            self.heads[node] = head

    @property
    def formula_ids(self) -> dict[str, list[str]]:
        return {n: list(f) for n, f in self._formula_ids.items()}

    @property
    def num_formulas(self) -> dict[str, int]:
        return dict(self._num_formulas)

    def _verify_default_bias(self) -> dict[str, bool]:
        verified = {}
        for node in _NODE_ORDER:
            fids = self._formula_ids[node]
            default_fid = _DEFAULT_ANCHOR[node]
            default_idx = fids.index(default_fid)
            bias_tensor = self.heads[node].bias.data
            # Default index should have the highest bias value in that head
            verified[node] = bool(bias_tensor.argmax().item() == default_idx)
        return verified

    def forward(self, attrs: torch.Tensor) -> dict:
        """Forward pass.

        Parameters
        ----------
        attrs: ``[B, attr_dim]`` float tensor of static basin attributes.

        Returns
        -------
        dict with keys:
            ``logits``, ``weights``, ``selected``, ``formula_ids``,
            ``entropy_<node>``, ``max_weight_<node>``.
        """
        is_training = self.training

        logits: dict[str, torch.Tensor] = {}
        weights: dict[str, torch.Tensor] = {}
        selected: dict[str, torch.Tensor] = {}
        entropy: dict[str, torch.Tensor] = {}
        max_weight: dict[str, torch.Tensor] = {}

        for node in _NODE_ORDER:
            head = self.heads[node]
            raw = head(attrs)  # [B, N]
            logits[node] = raw / self.temperature

            soft_probs = F.softmax(logits[node], dim=-1)
            n = soft_probs.shape[-1]

            if node == "recharge":
                _route_hard(soft_probs, n, is_training, weights, selected, node)
            else:
                if is_training:
                    weights[node] = soft_probs
                    selected[node] = soft_probs.argmax(dim=-1)
                elif self.hard_eval:
                    _route_hard(soft_probs, n, False, weights, selected, node)
                else:
                    weights[node] = soft_probs
                    selected[node] = soft_probs.argmax(dim=-1)

            p_safe = soft_probs.clamp(min=1e-8)
            H = -(p_safe * p_safe.log()).sum(dim=-1)
            H_max = math.log(n) if n > 1 else 1.0
            entropy[node] = H / H_max
            max_weight[node] = soft_probs.max(dim=-1).values

        result: dict = {
            "logits": dict(logits),
            "weights": dict(weights),
            "selected": dict(selected),
            "formula_ids": {n: list(f) for n, f in self._formula_ids.items()},
        }
        for node in _NODE_ORDER:
            result[f"entropy_{node}"] = entropy[node]
            result[f"max_weight_{node}"] = max_weight[node]

        return result


def _route_hard(
    soft_probs: torch.Tensor,
    n: int,
    is_training: bool,
    weights: dict[str, torch.Tensor],
    selected: dict[str, torch.Tensor],
    node: str,
) -> None:
    hard_onehot = F.one_hot(soft_probs.argmax(dim=-1), num_classes=n).float()
    selected[node] = hard_onehot.argmax(dim=-1)
    if is_training:
        weights[node] = hard_onehot + soft_probs - soft_probs.detach()
    else:
        weights[node] = hard_onehot
