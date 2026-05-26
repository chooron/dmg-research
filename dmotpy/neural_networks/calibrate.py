from typing import Union

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import qmc


def compute_nmul(ny: int, multiplier: int = 20) -> int:
    """Compute a power-of-two multi-start size from parameter count."""

    raw = max(multiplier * ny, 32)
    return int(2 ** np.ceil(np.log2(raw)))


class Calibrate(torch.nn.Module):
    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        num_basins: int = 100,
        num_start: int = 10,
        init_strategy: str = "lhs_logit",
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.name = "Calibrate"
        self.num_basins = num_basins
        self.ny = ny
        self.num_start = num_start
        self.device = device
        self.params = self._initialize_params(init_strategy)

    def _initialize_params(self, strategy: str) -> nn.Parameter:
        """Generate basin-wise initial parameters with LHS + logit."""

        if strategy != "lhs_logit":
            raise ValueError(f"Unsupported init_strategy: {strategy}")

        sampler = qmc.LatinHypercube(d=self.ny)
        basin_samples = []
        for _ in range(self.num_basins):
            basin_samples.append(sampler.random(n=self.num_start))

        sample_np = np.stack(basin_samples, axis=0)
        u = torch.from_numpy(sample_np).float().to(self.device).transpose(1, 2)
        u = u * 0.9 + 0.05
        init_val = torch.log(u / (1 - u))

        print(
            "[Calibrate] Initialized with per-basin Latin Hypercube Sampling "
            f"(LHS) + Logit With {self.num_start} Starts."
        )
        return nn.Parameter(init_val)

    @classmethod
    def build_by_config(cls, config: dict, device: str = "cpu"):
        n_basins = config.get("num_basins", 559)
        init_strat = config.get("init_strategy", "lhs_logit")
        ny = config["ny"]
        nmul_cfg = config.get("nmul", 16)
        num_start = compute_nmul(ny, multiplier=nmul_cfg)

        return cls(
            nx=config["nx2"],
            ny=ny,
            num_basins=n_basins,
            num_start=num_start,
            init_strategy=init_strat,
            device=device,
        )

    def forward(
        self,
        x: dict[str, torch.Tensor],
    ) -> tuple[Union[None, torch.Tensor], torch.Tensor]:
        batch_indices = x["batch_sample"]
        cur_params = self.params[batch_indices]
        return None, torch.sigmoid(cur_params)
