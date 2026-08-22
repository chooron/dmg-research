"""Experimental model classes for interception candidates E (bounded linear
cosine) and F (bounded logistic cosine), each under PET-cap semantics
S0 (production-style), S1 (V1 independent loss with PET cap) and
S2 (independent loss without interception PET cap).

The classes reuse the exact network interface of production
``LearnedWeightMopex``: same 12-slot parameter order (the alpha slot, index 8,
now carries ``kappa``; the is_time slot, index 9, carries ``phi``), same output
dimension, same learnable-parameter count, same gate initialization, same
official descale path (sigmoid -> hydrodl2 ``change_param_range``).  Only the
interception seasonal gate and the PET-cap semantics differ.

``kappa`` range: candidate E uses [0, 1] (unchanged bounds); candidate F uses
[0, KAPPA_MAX] via an override of the alpha-slot bounds (still the official
sigmoid -> change_param_range transform).
"""
from __future__ import annotations

from typing import Any
import torch
import torch.nn.functional as F
from project.flexmopex.models import mopex_core_candidates as cand
from project.flexmopex.models.base_mopex import MOPEX_PARAMS_BOUNDS
from project.flexmopex.models.learned_weight_mopex import LearnedWeightMopex
from project.flexmopex.models.parameter_nets import LearnedStructureNet

_SEMANTICS_TO_STEP = {
    "S0": {"pet_cap": True, "pet_independent": False},
    "S1": {"pet_cap": True, "pet_independent": True},
    "S2": {"pet_cap": False, "pet_independent": True},
}

class SensitivityReweightFunction(torch.autograd.Function):
    """Process-wise, basin-wise sensitivity reweighting of the structural-gate fit gradient.

    For each process column p independently:
      1. s[:, p] = abs(g[:, p]).detach()
      2. a_raw[:, p] = s[:, p] / (mean(s[:, p]) + eps)
      3. a_cap[:, p] = min(a_raw[:, p], cap)
      4. a[:, p] = a_cap[:, p] / (mean(a_cap[:, p]) + eps)
      5. g_tmp[:, p] = a[:, p] * g[:, p]
      6. g_tilde[:, p] = g_tmp[:, p] * (mean(abs(g[:, p])) + eps) / (mean(abs(g_tmp[:, p])) + eps)
    """

    @staticmethod
    def forward(ctx, weights: torch.Tensor, cap: float = 5.0, eps: float = 1e-12) -> torch.Tensor:
        ctx.cap = float(cap)
        ctx.eps = float(eps)
        return weights.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        cap = ctx.cap
        eps = ctx.eps
        g = grad_output  # [n_grid, n_proc]

        s = torch.abs(g)  # [n_grid, n_proc]
        s_mean = torch.mean(s, dim=0, keepdim=True)  # [1, n_proc]

        a_raw = s / (s_mean + eps)
        a_cap = torch.clamp(a_raw, max=cap)
        a_cap_mean = torch.mean(a_cap, dim=0, keepdim=True)
        a = a_cap / (a_cap_mean + eps)

        g_tmp = a * g
        g_tmp_abs_mean = torch.mean(torch.abs(g_tmp), dim=0, keepdim=True)
        scale = (s_mean + eps) / (g_tmp_abs_mean + eps)
        g_tilde = g_tmp * scale

        is_zero = (s_mean < eps)
        g_tilde = torch.where(is_zero, g, g_tilde)
        return g_tilde, None, None


def reweight_fit_gradient(weights: torch.Tensor, cap: float = 5.0, eps: float = 1e-12) -> torch.Tensor:
    return SensitivityReweightFunction.apply(weights, cap, eps)
class DirectionBalancedSensitivityReweightFunction(torch.autograd.Function):
    """Direction-balanced + sensitivity-weighted fit gradient aggregation (R11).

    For each process column p independently within the current basin batch:
      1. Partition non-zero basins into G_ON (g < 0) and G_OFF (g > 0).
      2. Within each non-empty group G:
           s_G = abs(g_G).detach()
           r_raw_G = s_G / (mean(s_G) + eps)
           r_cap_G = min(r_raw_G, cap)
           r_G = r_cap_G / (mean(r_cap_G) + eps)
      3. Direction balance:
           If both G_ON and G_OFF non-empty:
             b_ON = N / (2 * N_ON)
             b_OFF = N / (2 * N_OFF)
           Else:
             b = 1.0
      4. Combine and rescale:
           g_tmp = b * r * g
           scale = (mean(abs(g)) + eps) / (mean(abs(g_tmp)) + eps)
           g_tilde = g_tmp * scale
    """

    @staticmethod
    def forward(ctx, weights: torch.Tensor, cap: float = 5.0, eps: float = 1e-12) -> torch.Tensor:
        ctx.cap = float(cap)
        ctx.eps = float(eps)
        return weights.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        cap = ctx.cap
        eps = ctx.eps
        g = grad_output  # [N, P]
        N, P = g.shape
        g_tilde = torch.empty_like(g)

        for p in range(P):
            gp = g[:, p]
            s_all_mean = torch.mean(torch.abs(gp))
            if s_all_mean < eps:
                g_tilde[:, p] = gp
                continue

            mask_on = gp < 0
            mask_off = gp > 0
            n_on = int(mask_on.sum().item())
            n_off = int(mask_off.sum().item())

            gtmp_p = torch.zeros_like(gp)

            if n_on > 0 and n_off > 0:
                b_on = float(N) / (2.0 * float(n_on))
                b_off = float(N) / (2.0 * float(n_off))

                # ON group
                s_on = torch.abs(gp[mask_on])
                s_on_mean = torch.mean(s_on) + eps
                r_raw_on = s_on / s_on_mean
                r_cap_on = torch.clamp(r_raw_on, max=cap)
                r_on = r_cap_on / (torch.mean(r_cap_on) + eps)
                gtmp_p[mask_on] = b_on * r_on * gp[mask_on]

                # OFF group
                s_off = torch.abs(gp[mask_off])
                s_off_mean = torch.mean(s_off) + eps
                r_raw_off = s_off / s_off_mean
                r_cap_off = torch.clamp(r_raw_off, max=cap)
                r_off = r_cap_off / (torch.mean(r_cap_off) + eps)
                gtmp_p[mask_off] = b_off * r_off * gp[mask_off]

            elif n_on > 0:
                s_on = torch.abs(gp[mask_on])
                s_on_mean = torch.mean(s_on) + eps
                r_raw_on = s_on / s_on_mean
                r_cap_on = torch.clamp(r_raw_on, max=cap)
                r_on = r_cap_on / (torch.mean(r_cap_on) + eps)
                gtmp_p[mask_on] = 1.0 * r_on * gp[mask_on]

            elif n_off > 0:
                s_off = torch.abs(gp[mask_off])
                s_off_mean = torch.mean(s_off) + eps
                r_raw_off = s_off / s_off_mean
                r_cap_off = torch.clamp(r_raw_off, max=cap)
                r_off = r_cap_off / (torch.mean(r_cap_off) + eps)
                gtmp_p[mask_off] = 1.0 * r_off * gp[mask_off]

            gtmp_abs_mean = torch.mean(torch.abs(gtmp_p))
            scale = (s_all_mean + eps) / (gtmp_abs_mean + eps)
            g_tilde[:, p] = gtmp_p * scale

        return g_tilde, None, None


def direction_balanced_reweight_fit_gradient(
    weights: torch.Tensor, cap: float = 5.0, eps: float = 1e-12
) -> torch.Tensor:
    return DirectionBalancedSensitivityReweightFunction.apply(weights, cap, eps)
class SoftDenoisedSensitivityReweightFunction(torch.autograd.Function):
    """Soft-denoised sensitivity-reweighted fit gradient aggregation (R12).

    For each process column p independently within the current basin batch:
      1. m = abs(g[:, p]).detach()
      2. tau = median(m).detach()
      3. c = m^2 / (m^2 + tau^2 + eps)
      4. r_raw = m / (mean(m) + eps)
         r_cap = min(r_raw, cap)
         q = c * r_cap
      5. a = q / (mean(q) + eps)
         g_tmp = a * g[:, p]
      6. scale = (mean(abs(g[:, p])) + eps) / (mean(abs(g_tmp)) + eps)
         g_tilde = g_tmp * scale
    """

    @staticmethod
    def forward(ctx, weights: torch.Tensor, cap: float = 5.0, eps: float = 1e-12) -> torch.Tensor:
        ctx.cap = float(cap)
        ctx.eps = float(eps)
        return weights.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None, None]:
        cap = ctx.cap
        eps = ctx.eps
        g = grad_output  # [N, P]
        N, P = g.shape
        g_tilde = torch.empty_like(g)

        for p in range(P):
            gp = g[:, p]
            m = torch.abs(gp)
            m_mean = torch.mean(m)
            if m_mean < eps:
                g_tilde[:, p] = gp
                continue

            tau = torch.median(m).detach()
            c = (m ** 2) / (m ** 2 + tau ** 2 + eps)

            r_raw = m / (m_mean + eps)
            r_cap = torch.clamp(r_raw, max=cap)
            q = c * r_cap

            q_mean = torch.mean(q) + eps
            a = q / q_mean

            gtmp_p = a * gp
            gtmp_abs_mean = torch.mean(torch.abs(gtmp_p)) + eps
            scale = (m_mean + eps) / gtmp_abs_mean
            g_tilde[:, p] = gtmp_p * scale

        return g_tilde, None, None


def soft_denoised_reweight_fit_gradient(
    weights: torch.Tensor, cap: float = 5.0, eps: float = 1e-12
) -> torch.Tensor:
    return SoftDenoisedSensitivityReweightFunction.apply(weights, cap, eps)


class _CandidateBase(LearnedWeightMopex):
    """Shared machinery: pick the step function from config interception_semantics."""

    _season_mode: str = "linear"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(config, device)
        semantics = str(self.config.get("interception_semantics", "S0")).upper()
        if semantics not in _SEMANTICS_TO_STEP:
            raise ValueError(
                f"Unknown interception_semantics {semantics!r}; expected S0/S1/S2."
            )
        self.interception_semantics = semantics
        kwargs = _SEMANTICS_TO_STEP[semantics]
        step = cand._make_step(self._season_mode, **kwargs)
        self.step_fn = self._compile_step(step)
        self.freeze_wint = bool(self.config.get("freeze_wint", False))
        removed = self.config.get("removed_processes", self.config.get("removed_process", []))
        if isinstance(removed, str):
            removed = [removed]
        self.removed_processes = {str(name) for name in (removed or [])}
        unknown_removed = self.removed_processes - set(self.weight_names)
        if unknown_removed:
            raise ValueError(f"Unknown removed process names: {sorted(unknown_removed)}")
        # Full-process parameter warm-up (formal method):
        #   structure_warmup_epochs=N (default 0 = off): during training epochs
        #   1..N the effective structural gates are forced to exactly 1 (a
        #   detached forward override), so the hydrologic parameter network
        #   first adapts to the complete process structure.  Gate logits are
        #   NOT driven toward 1 and the gate head receives no gradient (the
        #   loss does not depend on the logits while the override is active),
        #   keeping the underlying gate state near-neutral for release.  From
        #   epoch N+1 the override is removed and ordinary joint parameter +
        #   structure training resumes.  Eval-mode forwards always use the
        #   learned softmax gates, so warm-up checkpoints report the raw/learned
        #   gate probabilities.  The epoch is pushed by the trainer via
        #   set_current_epoch (WarmupTrainer).
        warmup = self.config.get("structure_warmup_epochs", 0)
        warmup = 0 if warmup is None else int(warmup)
        if warmup < 0:
            raise ValueError(f"structure_warmup_epochs must be >= 0, got {warmup}")
        self.structure_warmup_epochs = warmup
        # Delayed gate-AIC gradient exposure (formal method):
        #   gate_aic_delay_epochs=N (default 0 = off): during training epochs
        #   1..N the AIC/complexity term keeps its exact reported value (the
        #   w_* outputs passed to the loss are value-identical), but those
        #   outputs are detached, so the AIC gradient cannot flow through the
        #   structural-gate/structure-network path.  Predictive-fit gradients
        #   through all four gates remain fully active (they flow via the
        #   streamflow graph, which is untouched).  From epoch N+1 the
        #   canonical full (fit + AIC) gradient resumes.  The schedule applies
        #   uniformly to all four structural processes (no interception-only
        #   exception).  The epoch is pushed by the trainer via
        #   set_current_epoch (WarmupTrainer).
        delay = self.config.get("gate_aic_delay_epochs", 0)
        delay = 0 if delay is None else int(delay)
        if delay < 0:
            raise ValueError(f"gate_aic_delay_epochs must be >= 0, got {delay}")
        self.gate_aic_delay_epochs = delay
        # Process-wise sensitivity reweighting of fit gradient into gate network (R10/R11/R12):
        #   reweight_gate_fit_gradients=bool (default False = off)
        #   direction_balanced_gate_gradients=bool (default False = off, R11 mode)
        #   soft_denoised_gate_gradients=bool (default False = off, R12 mode)
        #   reweight_gate_mode=str ("none", "sensitivity", "direction_balanced", "soft_denoised")
        #   reweight_gate_cap=float (default 5.0, upper cap on relative sensitivity)
        mode = str(self.config.get("reweight_gate_mode", "")).lower()
        soft_den = bool(self.config.get("soft_denoised_gate_gradients", False))
        dir_bal = bool(self.config.get("direction_balanced_gate_gradients", False))
        reweight = bool(self.config.get("reweight_gate_fit_gradients", False))

        if soft_den or mode in ("soft_denoised", "soft_denoising", "r12"):
            self.reweight_gate_mode = "soft_denoised"
        elif dir_bal or mode == "direction_balanced":
            self.reweight_gate_mode = "direction_balanced"
        elif reweight or mode == "sensitivity":
            self.reweight_gate_mode = "sensitivity"
        else:
            self.reweight_gate_mode = "none"

        self.reweight_gate_fit_gradients = (self.reweight_gate_mode != "none")
        cap = self.config.get("reweight_gate_cap", 5.0)
        cap = 5.0 if cap is None else float(cap)
        if cap <= 0:
            raise ValueError(f"reweight_gate_cap must be > 0, got {cap}")
        self.reweight_gate_cap = cap
        # Counterfactual structural supervision (R15):
        #   counterfactual_supervision=bool (default False = off)
        #   When active during training, weights_head receives training gradients ONLY
        #   from L_CF (detached BCE target). The direct predictive-fit -> weights_head
        #   and AIC -> weights_head gradient paths are completely detached in the physics loop,
        #   while forward gate values and non-gate gradients remain 100% value-identical.
        self.counterfactual_supervision = bool(self.config.get("counterfactual_supervision", False))
        self.current_epoch = 0

    def set_current_epoch(self, epoch: int) -> None:
        """Trainer hook: current 1-based training epoch (MyTrainer numbering)."""
        self.current_epoch = int(epoch)

    @property
    def _gate_aic_mask_active(self) -> bool:
        return (
            self.gate_aic_delay_epochs > 0
            and self.training
            and self.current_epoch <= self.gate_aic_delay_epochs
        )

    def forward(
        self,
        x_dict: dict[str, torch.Tensor],
        parameters: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Byte-identical to ``LearnedWeightMopex.forward`` except that during
        the gate-AIC delay window or counterfactual supervision mode the ``w_*``
        outputs (used by the loss for the AIC term only) are detached.
        """
        mopex_params = self._descale_mopex_params(parameters["params"])
        routing_params = self._descale_routing_params(parameters["gamma_uh"])
        weights_on = self._structure_weights(parameters["weights"])
        weights_on_fit = weights_on
        if self.training:
            if self.counterfactual_supervision:
                # R15: Block direct fit gradient to weights_head entirely.
                # Forward values are identical, but weights_on is detached for the physics loop.
                weights_on_fit = weights_on.detach()
            elif self.reweight_gate_mode == "soft_denoised":
                weights_on_fit = soft_denoised_reweight_fit_gradient(
                    weights_on, cap=self.reweight_gate_cap
                )
            elif self.reweight_gate_mode == "direction_balanced":
                weights_on_fit = direction_balanced_reweight_fit_gradient(
                    weights_on, cap=self.reweight_gate_cap
                )
            elif self.reweight_gate_mode == "sensitivity":
                weights_on_fit = reweight_fit_gradient(
                    weights_on, cap=self.reweight_gate_cap
                )
        P, T, PET, doy, n_steps, n_grid = self._prepare_forcings(x_dict)
        Q_mopex = self._run_weighted_loop(
            P, T, PET, doy, mopex_params, weights_on_fit, n_steps, n_grid
        )
        Qrouted = self._apply_routing(Q_mopex.mean(-1), routing_params)
        result = {"streamflow": Qrouted}
        w_out = self._weight_outputs(weights_on, Q_mopex.shape[0])
        if self._gate_aic_mask_active or (self.counterfactual_supervision and self.training):
            # Cut direct AIC -> gate-logit gradient path (value-identical).
            w_out = {name: t.detach() for name, t in w_out.items()}
        result.update(w_out)
        return result

    def _structure_weights(self, raw_weights: torch.Tensor) -> torch.Tensor:
        logits = raw_weights.view(raw_weights.shape[0], len(self.weight_names), 2)
        logits = torch.clamp(logits, min=-10.0, max=10.0)
        if self.training:
            probs = F.gumbel_softmax(
                logits, tau=self.structure_tau, hard=False, dim=-1
            )
        else:
            probs = F.softmax(logits, dim=-1)
        warmup_active = (
            self.structure_warmup_epochs > 0
            and self.training
            and self.current_epoch <= self.structure_warmup_epochs
        )
        if warmup_active:
            # Full-process warm-up: effective gates = 1 (detached constant; no
            # gradient path to the gate logits -> gate head does not update).
            probs = torch.ones_like(probs)
        elif self.freeze_wint:
            frozen = torch.full_like(probs[:, 1:2, :], 0.5)
            probs = torch.cat([probs[:, 0:1, :], frozen, probs[:, 2:, :]], dim=1)
        if self.removed_processes:
            process_mask = torch.ones_like(probs)
            for index, name in enumerate(self.weight_names):
                if name in self.removed_processes:
                    process_mask[:, index, :] = torch.tensor((1.0, 0.0), device=probs.device, dtype=probs.dtype)
            probs = probs * process_mask
        return probs[..., 1]


class LearnedWeightMopexE(_CandidateBase):
    """Candidate E: bounded linear cosine gate, kappa in [0, 1]."""

    _season_mode = "linear"

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(config, device)
        self.name = "LearnedWeightMopexE"


class LearnedWeightMopexF(_CandidateBase):
    """Candidate F: bounded logistic cosine gate, kappa in [0, KAPPA_MAX]."""

    _season_mode = "logistic"
    # alpha slot keeps its index but its valid range becomes [0, KAPPA_MAX]
    # (official transform: sigmoid -> change_param_range with these bounds).
    param_bounds = {**MOPEX_PARAMS_BOUNDS, "alpha": [0.0, cand.KAPPA_MAX]}

    def __init__(
        self,
        config: dict[str, Any] | None = None,
        device: str | torch.device | None = None,
    ) -> None:
        super().__init__(config, device)
        self.name = "LearnedWeightMopexF"


class LearnedStructureNetCF(LearnedStructureNet):
    """LearnedStructureNet with detached backbone input to weights_head (R15).

    Guarantees that L_CF gradient to weights_head does not update the shared backbone.
    """

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        shared = self.backbone(x["c_nn_norm"])
        return {
            "params": self.heads["params"](shared),
            "gamma_uh": self.heads["gamma_uh"](shared),
            "weights": self.heads["weights"](shared.detach()),
        }


class LearnedStructureNetDirectAttr(LearnedStructureNet):
    """LearnedStructureNet with Direct-Attribute Structure Head (R18).

    Architecture:
      x35_norm -> shared backbone (128-D) -> params_head (192) / gamma_head (2)
      x35_norm -> weights_head (Linear(35, 8)) -> 8 structure logits

    Eliminates shared-backbone mediation for structural gate decisions.
    L_CF gradient updates only the direct-attribute weights_head and has
    strictly zero gradient path to the shared backbone by construction.
    """

    def __init__(
        self,
        input_dim: int = 27,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        nmul: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            nmul=nmul,
            device=device,
        )
        # Replace weights_head with direct linear projection: input_dim (35) -> 8
        self.heads["weights"] = torch.nn.Linear(input_dim, 8)
        torch.nn.init.normal_(self.heads["weights"].weight, mean=0.0, std=0.001)
        if self.heads["weights"].bias is not None:
            torch.nn.init.constant_(self.heads["weights"].bias, 0.0)
        self.to(device)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        attrs = x["c_nn_norm"]
        shared = self.backbone(attrs)
        return {
            "params": self.heads["params"](shared),
            "gamma_uh": self.heads["gamma_uh"](shared),
            "weights": self.heads["weights"](attrs),  # Direct from raw 35-D static attributes!
        }

    def get_structure_logits(self, attrs: torch.Tensor) -> torch.Tensor:
        """Compute 8 structure gate logits directly from static basin attributes."""
        return self.heads["weights"](attrs)

    def structure_parameters(self):
        """Iterator over parameters belonging to the structure prediction branch."""
        return self.heads["weights"].parameters()


class LearnedStructureNetHybridEncoder(LearnedStructureNet):
    """LearnedStructureNet with Hybrid Dedicated Structure Encoder (R18).

    Architecture:
      Hydrologic branch: x35_norm -> shared backbone (128-D) -> params_head (192) / gamma_head (2)
      Structure branch : [x35_norm, stopgrad(h128)] -> dedicated MLP (163 -> 128 -> 64 -> 8) -> 8 logits

    Guarantees that L_CF gradient updates the dedicated structure MLP only,
    with strictly zero gradient path to the shared hydrologic backbone.
    """

    def __init__(
        self,
        input_dim: int = 27,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        nmul: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            nmul=nmul,
            device=device,
        )
        # Dedicated nonlinear structure encoder: (35 + 128 = 163) -> 128 -> 64 -> 8
        struct_in_dim = input_dim + hidden_dim
        self.structure_encoder = torch.nn.Sequential(
            torch.nn.Linear(struct_in_dim, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 64),
            torch.nn.Tanh(),
            torch.nn.Linear(64, 8),
        )
        # Initialize hidden layers with xavier_uniform, final output layer with normal(0, 0.001)
        torch.nn.init.xavier_uniform_(self.structure_encoder[0].weight)
        torch.nn.init.constant_(self.structure_encoder[0].bias, 0.0)
        torch.nn.init.xavier_uniform_(self.structure_encoder[2].weight)
        torch.nn.init.constant_(self.structure_encoder[2].bias, 0.0)
        torch.nn.init.normal_(self.structure_encoder[4].weight, mean=0.0, std=0.001)
        torch.nn.init.constant_(self.structure_encoder[4].bias, 0.0)
        self.to(device)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        attrs = x["c_nn_norm"]
        shared = self.backbone(attrs)
        # Hybrid input: concatenate normalized 35-D static attributes with detached 128-D hydrologic representation
        struct_input = torch.cat([attrs, shared.detach()], dim=-1)  # [B, 163]
        weights = self.structure_encoder(struct_input)             # [B, 8]
        return {
            "params": self.heads["params"](shared),
            "gamma_uh": self.heads["gamma_uh"](shared),
            "weights": weights,
        }

    def get_structure_logits(self, attrs: torch.Tensor) -> torch.Tensor:
        """Compute 8 structure gate logits from hybrid [attrs, stopgrad(h128)] input (legacy R18)."""
        with torch.no_grad():
            shared_detached = self.backbone(attrs)
        struct_input = torch.cat([attrs, shared_detached], dim=-1)
        return self.structure_encoder(struct_input)

    def structure_parameters(self):
        """Iterator over parameters belonging to the structure prediction branch."""
        return self.structure_encoder.parameters()


class LearnedStructureNetPureAttrEncoder(LearnedStructureNet):
    """LearnedStructureNet with Pure-Attribute Dedicated Nonlinear Structure Encoder (Canonical).

    Architecture:
      Hydrologic branch: x35_norm -> shared backbone (128-D) -> params_head (192) / gamma_head (2)
      Structure branch : x35_norm -> dedicated MLP (35 -> 128 -> 128 -> 8) -> 8 logits

    Zero Shared-Backbone Leakage:
      The structure encoder directly ingests normalized basin attributes x35_norm without
      detouring through or concatenating stopgrad(h128). L_CF gradient updates the dedicated
      structure MLP only, with strictly zero gradient path to the shared hydrologic backbone.
    """

    def __init__(
        self,
        input_dim: int = 27,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        nmul: int = 1,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            nmul=nmul,
            device=device,
        )
        # Dedicated pure-attribute nonlinear structure encoder: input_dim (35) -> 128 -> 128 -> 8
        self.structure_encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 128),
            torch.nn.Tanh(),
            torch.nn.Linear(128, 8),
        )
        # Initialize hidden layers with xavier_uniform, final output layer with normal(0, 0.001)
        torch.nn.init.xavier_uniform_(self.structure_encoder[0].weight)
        torch.nn.init.constant_(self.structure_encoder[0].bias, 0.0)
        torch.nn.init.xavier_uniform_(self.structure_encoder[2].weight)
        torch.nn.init.constant_(self.structure_encoder[2].bias, 0.0)
        torch.nn.init.normal_(self.structure_encoder[4].weight, mean=0.0, std=0.001)
        torch.nn.init.constant_(self.structure_encoder[4].bias, 0.0)
        self.to(device)

    def forward(self, x: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        attrs = x["c_nn_norm"]
        shared = self.backbone(attrs)
        # Pure-attribute input: direct from raw normalized static attributes
        weights = self.structure_encoder(attrs)  # [B, 8]
        return {
            "params": self.heads["params"](shared),
            "gamma_uh": self.heads["gamma_uh"](shared),
            "weights": weights,
        }

    def get_structure_logits(self, attrs: torch.Tensor) -> torch.Tensor:
        """Compute 8 structure gate logits directly from normalized static basin attributes (canonical)."""
        return self.structure_encoder(attrs)

    def structure_parameters(self):
        """Iterator over parameters belonging to the structure prediction branch."""
        return self.structure_encoder.parameters()
