"""Independent full-covariance CMA-ES distributions batched over basin/start."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class BatchedCMAState:
    mean: torch.Tensor
    covariance: torch.Tensor
    factor: torch.Tensor
    sigma: torch.Tensor
    path_sigma: torch.Tensor
    path_covariance: torch.Tensor
    best_fitness: torch.Tensor
    best_candidate: torch.Tensor
    generation: int = 0


class BatchedCMAES:
    """Run independent active CMA-ES distributions in one tensor batch."""

    def __init__(self, units: int, dimension: int, population: int, *, device: torch.device,
                 stdev_init: float = 0.20, active: bool = True) -> None:
        if units < 1 or dimension < 1 or population < 4:
            raise ValueError("units and dimension must be positive; population must be at least four")
        self.units, self.dimension, self.population = units, dimension, population
        self.device, self.active = device, active
        self.mu = population // 2
        raw = torch.log(torch.tensor((population + 1) / 2, dtype=torch.float64, device=device))
        raw = raw - torch.log(torch.arange(population, dtype=torch.float64, device=device) + 1)
        positive, negative = raw[:self.mu], raw[self.mu:]
        self.mu_eff = positive.sum().square() / positive.square().sum()
        self.c_sigma = (self.mu_eff + 2.0) / (dimension + self.mu_eff + 3.0)
        self.damp_sigma = 1 + 2 * max(0.0, float(torch.sqrt((self.mu_eff - 1) / (dimension + 1)) - 1)) + self.c_sigma
        self.c_c = (4 + self.mu_eff / dimension) / (dimension + 4 + 2 * self.mu_eff / dimension)
        self.c_1 = min(1.0, population / 6.0) * 2 / ((dimension + 1.3) ** 2 + self.mu_eff)
        self.c_mu = min(1 - self.c_1, 2 * ((0.25 + self.mu_eff - 2 + 1 / self.mu_eff) / ((dimension + 2) ** 2 + self.mu_eff)))
        self.discount_sigma = torch.sqrt(self.c_sigma * (2 - self.c_sigma) * self.mu_eff)
        self.discount_covariance = torch.sqrt(self.c_c * (2 - self.c_c) * self.mu_eff)
        positive = positive / positive.sum()
        if active:
            mu_eff_neg = negative.sum().square() / negative.square().sum()
            alpha = min(1 + self.c_1 / self.c_mu, 1 + 2 * mu_eff_neg / (self.mu_eff + 2),
                        (1 - self.c_mu - self.c_1) / (dimension * self.c_mu))
            negative = alpha * negative / negative.abs().sum()
        else:
            negative = torch.zeros_like(negative)
        self.weights = torch.cat([positive, negative])
        self.expected_norm = math.sqrt(dimension) * (1 - 1 / (4 * dimension) + 1 / (21 * dimension ** 2))
        self.decompose_frequency = max(1, int(math.floor(1 / (10 * dimension * float(self.c_1 + self.c_mu)))))
        eye = torch.eye(dimension, dtype=torch.float64, device=device).expand(units, -1, -1).clone()
        self.state = BatchedCMAState(
            mean=torch.zeros((units, dimension), dtype=torch.float64, device=device),
            covariance=eye.clone(), factor=eye,
            sigma=torch.full((units,), stdev_init, dtype=torch.float64, device=device),
            path_sigma=torch.zeros((units, dimension), dtype=torch.float64, device=device),
            path_covariance=torch.zeros((units, dimension), dtype=torch.float64, device=device),
            best_fitness=torch.full((units,), -torch.inf, dtype=torch.float64, device=device),
            best_candidate=torch.zeros((units, dimension), dtype=torch.float64, device=device),
        )

    def set_centers(self, centers: torch.Tensor) -> None:
        if centers.shape != (self.units, self.dimension):
            raise ValueError("center shape mismatch")
        self.state.mean.copy_(centers.to(self.device, dtype=torch.float64))
        self.state.best_candidate.copy_(self.state.mean)

    def ask(self, standard_normals: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if standard_normals.shape != (self.units, self.population, self.dimension):
            raise ValueError("standard-normal shape mismatch")
        z = standard_normals.to(self.device, dtype=torch.float64)
        shaped = torch.einsum("uij,upj->upi", self.state.factor, z)
        candidates = self.state.mean[:, None, :] + self.state.sigma[:, None, None] * shaped
        return z, candidates

    def tell(self, z: torch.Tensor, candidates: torch.Tensor, fitness: torch.Tensor) -> None:
        if fitness.shape != (self.units, self.population):
            raise ValueError("fitness shape mismatch")
        f = torch.nan_to_num(fitness.to(self.device, dtype=torch.float64), nan=-999.0, posinf=-999.0, neginf=-999.0)
        shaped = (candidates - self.state.mean[:, None, :]) / self.state.sigma[:, None, None].clamp_min(1e-20)
        order = torch.argsort(f, dim=1, descending=True)
        ranks = torch.empty_like(order)
        ranks.scatter_(1, order, torch.arange(self.population, device=self.device).expand(self.units, -1))
        assigned = self.weights[ranks]
        top = torch.topk(assigned, self.mu, dim=1)
        gather = top.indices[..., None].expand(-1, -1, self.dimension)
        local = (top.values[..., None] * z.gather(1, gather)).sum(1)
        selected = shaped.gather(1, gather)
        mean_step = (top.values[..., None] * selected).sum(1)
        self.state.mean.add_(self.state.sigma[:, None] * mean_step)
        self.state.path_sigma.mul_(1 - self.c_sigma).add_(self.discount_sigma * local)
        self.state.sigma.mul_(torch.exp((self.state.path_sigma.norm(dim=1) / self.expected_norm - 1) * self.c_sigma / self.damp_sigma))
        denominator = 1 - (1 - self.c_sigma) ** (2 * self.state.generation + 1)
        h_sigma = ((self.state.path_sigma.norm(dim=1).square() / denominator / self.dimension - 1) < 1 + 4 / (self.dimension + 1)).to(torch.float64)
        self.state.path_covariance.mul_(1 - self.c_c).add_(h_sigma[:, None] * self.discount_covariance * mean_step)
        covariance_weights = assigned
        if self.active:
            covariance_weights = torch.where(covariance_weights > 0, covariance_weights,
                                             self.dimension * covariance_weights / z.norm(dim=2).square().clamp_min(1e-23))
        c1a = self.c_1 * (1 - (1 - h_sigma.square()) * self.c_c * (2 - self.c_c))
        scaled_path = torch.sqrt(self.c_1 / (c1a + 1e-23))[:, None] * self.state.path_covariance
        rank_one = c1a[:, None, None] * (scaled_path[:, :, None] * scaled_path[:, None, :] - self.state.covariance)
        rank_mu = self.c_mu * ((covariance_weights[:, :, None, None] * shaped[:, :, :, None] * shaped[:, :, None, :]).sum(1) - self.weights.sum() * self.state.covariance)
        self.state.covariance.add_(rank_one + rank_mu)
        self.state.covariance.copy_(0.5 * (self.state.covariance + self.state.covariance.transpose(1, 2)))
        self.state.generation += 1
        if self.state.generation % self.decompose_frequency == 0:
            jitter = torch.eye(self.dimension, dtype=torch.float64, device=self.device)[None] * 1e-12
            try:
                self.state.factor.copy_(torch.linalg.cholesky(self.state.covariance + jitter))
            except RuntimeError:
                bad = torch.linalg.eigvalsh(self.state.covariance)[:, 0] <= 1e-12
                self.state.covariance[bad] = torch.eye(self.dimension, dtype=torch.float64, device=self.device)
                self.state.factor.copy_(torch.linalg.cholesky(self.state.covariance + jitter))
        best, index = f.max(dim=1)
        improved = best > self.state.best_fitness
        self.state.best_fitness[improved] = best[improved]
        self.state.best_candidate[improved] = candidates[torch.arange(self.units, device=self.device), index][improved]

    def state_dict(self) -> dict[str, Any]:
        return {"units": self.units, "dimension": self.dimension, "population": self.population,
                "state": {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                          for key, value in self.state.__dict__.items()}}

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if (payload["units"], payload["dimension"], payload["population"]) != (self.units, self.dimension, self.population):
            raise ValueError("checkpoint solver shape mismatch")
        for key, value in payload["state"].items():
            setattr(self.state, key, value.to(self.device) if isinstance(value, torch.Tensor) else value)
