from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import torch


def stable_hash(*parts: object) -> int:
    digest = hashlib.blake2b("|".join(map(str, parts)).encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little") & ((1 << 63) - 1)


def lhs_latent(n: int, d: int, seed: int, device: torch.device) -> torch.Tensor:
    """Deterministic stratified centers in normalized space, returned as logits."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    u = torch.empty((n, d), dtype=torch.float64)
    for j in range(d):
        u[:, j] = (torch.rand(n, generator=generator, dtype=torch.float64) + torch.randperm(n, generator=generator)) / n
    return torch.logit(u.clamp(1e-7, 1 - 1e-7)).to(device)


@dataclass
class CMAState:
    mean: torch.Tensor
    C: torch.Tensor
    A: torch.Tensor
    sigma: torch.Tensor
    p_sigma: torch.Tensor
    p_c: torch.Tensor
    best_fitness: torch.Tensor
    best_latent: torch.Tensor
    generation: int = 0


class BatchedCMAES:
    """Independent full-covariance Active CMA-ES distributions, batched over units.

    Formula defaults and active covariance reweighting are copied from the installed
    EvoTorch 0.6.1 `CMAES`; only the leading unit dimension is added.
    """

    def __init__(self, units: int, dimension: int, population: int, *, stdev_init: float,
                 active: bool, seed: int, device: str | torch.device = "cuda") -> None:
        self.units, self.dimension, self.population = int(units), int(dimension), int(population)
        self.mu = population // 2
        self.device = torch.device(device)
        self.active = bool(active)
        self.generator = torch.Generator(device=self.device).manual_seed(int(seed))
        raw = torch.log(torch.tensor((population + 1) / 2, dtype=torch.float64, device=self.device)) - torch.log(torch.arange(population, dtype=torch.float64, device=self.device) + 1)
        positive, negative = raw[:self.mu], raw[self.mu:]
        self.mu_eff = positive.sum().square() / positive.square().sum()
        self.c_m = 1.0
        self.c_sigma = (self.mu_eff + 2.0) / (dimension + self.mu_eff + 3.0)
        self.damp_sigma = 1 + 2 * max(0.0, float(torch.sqrt((self.mu_eff - 1) / (dimension + 1)) - 1)) + self.c_sigma
        self.c_c = (4 + self.mu_eff / dimension) / (dimension + (4 + 2 * self.mu_eff / dimension))
        self.c_1 = min(1.0, population / 6.0) * 2 / ((dimension + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * ((0.25 + self.mu_eff - 2 + 1 / self.mu_eff) / ((dimension + 2) ** 2 + self.mu_eff)))
        self.variance_discount_sigma = torch.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
        self.variance_discount_c = torch.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff)
        positive = positive / positive.sum()
        if active:
            mu_eff_neg = negative.sum().square() / negative.square().sum()
            alpha = min(1 + self.c_1 / self.c_mu, 1 + 2 * mu_eff_neg / (self.mu_eff + 2), (1 - self.c_mu - self.c_1) / (dimension * self.c_mu))
            negative = alpha * negative / negative.abs().sum()
        else:
            negative = torch.zeros_like(negative)
        self.weights = torch.cat([positive, negative])
        self.expected_norm = math.sqrt(dimension) * (1 - 1 / (4 * dimension) + 1 / (21 * dimension**2))
        self.decompose_frequency = max(1, int(math.floor(1 / (10 * dimension * float(self.c_1 + self.c_mu)))))
        eye = torch.eye(dimension, dtype=torch.float64, device=self.device).expand(units, -1, -1).clone()
        self.state = CMAState(torch.zeros((units, dimension), dtype=torch.float64, device=self.device), eye.clone(), eye,
                              torch.full((units,), stdev_init, dtype=torch.float64, device=self.device),
                              torch.zeros((units, dimension), dtype=torch.float64, device=self.device),
                              torch.zeros((units, dimension), dtype=torch.float64, device=self.device),
                              torch.full((units,), -torch.inf, dtype=torch.float64, device=self.device),
                              torch.zeros((units, dimension), dtype=torch.float64, device=self.device))

    def set_centers(self, centers: torch.Tensor) -> None:
        if centers.shape != self.state.mean.shape:
            raise ValueError(f"center shape {tuple(centers.shape)} != {tuple(self.state.mean.shape)}")
        self.state.mean.copy_(centers.to(self.device, torch.float64))
        self.state.best_latent.copy_(self.state.mean)

    def ask(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = torch.randn((self.units, self.population, self.dimension), generator=self.generator, dtype=torch.float64, device=self.device)
        y = torch.einsum("uij,upj->upi", self.state.A, z)
        x = self.state.mean[:, None, :] + self.state.sigma[:, None, None] * y
        return z, y, x

    def tell(self, z: torch.Tensor, y: torch.Tensor, x: torch.Tensor, fitness: torch.Tensor) -> None:
        """Update from maximisation fitness [unit,population]."""
        f = fitness.to(self.device, torch.float64)
        order = torch.argsort(f, dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(self.population, device=self.device).expand(self.units, -1))
        assigned = self.weights[ranks]
        top = torch.topk(assigned, self.mu, dim=1)
        gather = top.indices[..., None].expand(-1, -1, self.dimension)
        local = (top.values[..., None] * z.gather(1, gather)).sum(1)
        shaped = (top.values[..., None] * y.gather(1, gather)).sum(1)
        self.state.mean.add_(self.c_m * self.state.sigma[:, None] * shaped)
        self.state.p_sigma.mul_(1 - self.c_sigma).add_(self.variance_discount_sigma * local)
        exp_update = (self.state.p_sigma.norm(dim=1) / self.expected_norm - 1) * (self.c_sigma / self.damp_sigma)
        self.state.sigma.mul_(torch.exp(exp_update))
        denom = 1 - (1 - self.c_sigma) ** (2 * self.state.generation + 1)
        h_sig = (((self.state.p_sigma.norm(dim=1).square() / denom) / self.dimension - 1) < 1 + 4 / (self.dimension + 1)).to(torch.float64)
        self.state.p_c.mul_(1 - self.c_c).add_(h_sig[:, None] * self.variance_discount_c * shaped)
        cov_weights = assigned
        if self.active:
            cov_weights = torch.where(cov_weights > 0, cov_weights, self.dimension * cov_weights / z.norm(dim=2).square().clamp_min(1e-23))
        c1a = self.c_1 * (1 - (1 - h_sig.square()) * self.c_c * (2 - self.c_c))
        scaled_pc = torch.sqrt(self.c_1 / (c1a + 1e-23))[:, None] * self.state.p_c
        rank_one = c1a[:, None, None] * (scaled_pc[:, :, None] * scaled_pc[:, None, :] - self.state.C)
        rank_mu = self.c_mu * ((cov_weights[:, :, None, None] * y[:, :, :, None] * y[:, :, None, :]).sum(1) - self.weights.sum() * self.state.C)
        self.state.C.add_(rank_one + rank_mu)
        self.state.C.copy_(0.5 * (self.state.C + self.state.C.transpose(1, 2)))
        self.state.generation += 1
        if self.state.generation % self.decompose_frequency == 0:
            jitter = torch.eye(self.dimension, dtype=torch.float64, device=self.device)[None] * 1e-12
            try:
                self.state.A.copy_(torch.linalg.cholesky(self.state.C + jitter))
            except RuntimeError:
                # isolate failed solvers: reset only non-positive-definite units, retaining best state
                eig = torch.linalg.eigvalsh(self.state.C)
                bad = eig[:, 0] <= 1e-12
                self.state.C[bad] = torch.eye(self.dimension, dtype=torch.float64, device=self.device)
                self.state.A.copy_(torch.linalg.cholesky(self.state.C + jitter))
        best, index = f.max(dim=1)
        improved = best > self.state.best_fitness
        self.state.best_fitness[improved] = best[improved]
        self.state.best_latent[improved] = x[torch.arange(self.units, device=self.device), index][improved]

    def state_dict(self) -> dict[str, Any]:
        return {"units": self.units, "dimension": self.dimension, "population": self.population,
                "active": self.active, "generator_state": self.generator.get_state().cpu(),
                "state": {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in self.state.__dict__.items()}}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if (payload["units"], payload["dimension"], payload["population"]) != (self.units, self.dimension, self.population):
            raise ValueError("checkpoint solver shape mismatch")
        self.generator.set_state(payload["generator_state"].cpu())
        for key, value in payload["state"].items():
            setattr(self.state, key, value.to(self.device) if isinstance(value, torch.Tensor) else value)
