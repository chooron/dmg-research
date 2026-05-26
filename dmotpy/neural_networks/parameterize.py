from typing import Union

import torch
import torch.nn as nn


class Parameterize(torch.nn.Module):
    """Static-attribute to parameter prediction network."""

    def __init__(
        self,
        *,
        nx: int,
        ny: int,
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout_rate: float = 0.15,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        self.name = "Parameterize"
        self.ny = ny
        self.device = device

        layers = []
        in_size = nx
        for _ in range(num_layers):
            layers.append(nn.Linear(in_size, hidden_size))
            layers.append(nn.LayerNorm(hidden_size))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(p=dropout_rate))
            in_size = hidden_size

        layers.append(nn.Linear(hidden_size, ny))
        self.mlp = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    @classmethod
    def build_by_config(cls, config: dict, device: str = "cpu"):
        return cls(
            nx=config["nx2"],
            ny=config["ny"],
            hidden_size=config.get("hidden_size", 128),
            num_layers=config.get("num_layers", 2),
            dropout_rate=config.get("dropout_rate", 0.2),
            device=device,
        )

    def forward(
        self,
        x: dict[str, torch.Tensor],
    ) -> tuple[Union[None, torch.Tensor], torch.Tensor]:
        attr = x["c_nn_norm"]
        raw = self.mlp(attr)
        params = torch.sigmoid(raw).unsqueeze(-1)
        return None, params


def mc_dropout_inference(
    model: Parameterize,
    x: dict[str, torch.Tensor],
    n_samples: int = 100,
) -> dict[str, torch.Tensor]:
    """Run MC-dropout sampling and return parameter statistics."""

    model.train()
    samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            _, params = model(x)
            samples.append(params.squeeze(-1))

    samples = torch.stack(samples, dim=0)
    return {
        "mean": samples.mean(dim=0),
        "std": samples.std(dim=0),
        "p10": samples.quantile(0.1, dim=0),
        "p90": samples.quantile(0.9, dim=0),
        "samples": samples,
    }
