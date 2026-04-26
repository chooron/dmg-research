from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from project.bettermodel.implements.neural_networks.layers.ann import AnnModel
from project.bettermodel.implements.neural_networks.layers.hope import S4D


@dataclass(frozen=True)
class AblationSpec:
    name: str
    add_conv: bool
    norm: str
    dynamic_activation: str
    purpose: str


S4D_BENCHMARK_CFG = {
    "lr_min": 0.001,
    "lr": 0.01,
    "lr_dt": 0.0,
    "min_dt": 0.001,
    "max_dt": 1,
    "wd": 0.0,
    "d_state": 64,
    "cfr": 1.0,
    "cfi": 1.0,
    "use_gated": False,
    "out_activation": "glu",
}


def _apply_norm(norm: nn.Module, x: torch.Tensor, norm_type: str) -> torch.Tensor:
    if norm_type == "batch":
        return norm(x)
    if norm_type == "layer":
        return norm(x.transpose(-1, -2)).transpose(-1, -2)
    raise ValueError(f"Unsupported normalization type: {norm_type!r}")


def _normalized_activation(x: torch.Tensor, activation: str) -> torch.Tensor:
    if activation == "sigmoid":
        return torch.sigmoid(x)
    if activation == "softsign":
        return 0.5 * (F.softsign(x) + 1.0)
    raise ValueError(f"Unsupported dynamic activation: {activation!r}")


class ComponentAblationS4D(nn.Module):
    """S4D backbone with only conv/norm/output activation switched."""

    def __init__(
        self,
        *,
        input_size: int,
        output_size: int,
        hidden_size: int,
        dropout: float,
        n_layers: int,
        add_conv: bool,
        norm: str,
        conv_kernel_size: int = 5,
        cfg: dict | None = None,
    ) -> None:
        super().__init__()
        self.norm_type = norm
        self.add_conv = add_conv
        self.encoder = nn.Linear(input_size, hidden_size)
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        s4d_cfg = dict(S4D_BENCHMARK_CFG if cfg is None else cfg)
        for _ in range(n_layers):
            self.s4_layers.append(
                S4D(
                    hidden_size,
                    dropout=dropout,
                    transposed=True,
                    lr=min(s4d_cfg["lr_min"], s4d_cfg["lr"]),
                    d_state=s4d_cfg["d_state"],
                    dt_min=s4d_cfg["min_dt"],
                    dt_max=s4d_cfg["max_dt"],
                    lr_dt=s4d_cfg["lr_dt"],
                    cfr=s4d_cfg["cfr"],
                    cfi=s4d_cfg["cfi"],
                    wd=s4d_cfg["wd"],
                    out_activation=s4d_cfg["out_activation"],
                    use_gated=s4d_cfg["use_gated"],
                )
            )
            if norm == "batch":
                self.norms.append(nn.BatchNorm1d(hidden_size))
            elif norm == "layer":
                self.norms.append(nn.LayerNorm(hidden_size))
            else:
                raise ValueError(f"Unsupported normalization type: {norm!r}")
            self.dropouts.append(nn.Dropout(dropout))

        if add_conv:
            padding = conv_kernel_size // 2
            self.conv = nn.Conv1d(
                hidden_size,
                hidden_size,
                kernel_size=conv_kernel_size,
                padding=padding,
                groups=hidden_size,
                bias=False,
            )
        else:
            self.conv = nn.Identity()

        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = x.transpose(-1, -2)

        for layer, norm, dropout in zip(
            self.s4_layers,
            self.norms,
            self.dropouts,
        ):
            z = x
            z, _ = layer(z)
            z = dropout(z)
            x = _apply_norm(norm, z + x, self.norm_type)

        x = self.conv(x)
        x = x.transpose(-1, -2)
        return self.decoder(x)


class S5DAblationMlp(nn.Module):
    SPEC = AblationSpec(
        name="abstract",
        add_conv=False,
        norm="batch",
        dynamic_activation="sigmoid",
        purpose="base class",
    )

    def __init__(
        self,
        *,
        nx1: int,
        ny1: int,
        hiddeninv1: int,
        nx2: int,
        ny2: int,
        hiddeninv2: int,
        nmul: int,
        dr1: Optional[float] = 0.5,
        dr2: Optional[float] = 0.5,
        device: Optional[str] = "cpu",
    ) -> None:
        super().__init__()
        self.name = self.SPEC.name
        self.num_start = 1
        self.ny1 = ny1
        self.nmul = nmul
        self.dynamic_param_count = ny1 // nmul if nmul > 0 and ny1 % nmul == 0 else ny1
        self.hope_layer = ComponentAblationS4D(
            input_size=nx1,
            output_size=ny1,
            hidden_size=hiddeninv1,
            dropout=dr1 or 0.0,
            n_layers=4,
            add_conv=self.SPEC.add_conv,
            norm=self.SPEC.norm,
        )
        self.ann = AnnModel(
            nx=nx2,
            ny=ny2,
            hidden_size=hiddeninv2,
            dr=dr2,
        )
        self.to(torch.device(device or "cpu"))

    @classmethod
    def build_by_config(cls, config: dict, device: Optional[str] = "cpu"):
        return cls(
            nx1=config.get("nx1", config["nx"]),
            nx2=config["nx2"],
            ny1=config["ny1"],
            ny2=config["ny2"],
            hiddeninv1=config["hope_hidden_size"],
            hiddeninv2=config["mlp_hidden_size"],
            nmul=config.get("nmul", 1),
            dr1=config["hope_dropout"],
            dr2=config["mlp_dropout"],
            device=device,
        )

    def _dynamic_head(self, z1: torch.Tensor) -> torch.Tensor:
        raw = self.hope_layer(torch.permute(z1, (1, 0, 2))).permute(1, 0, 2)
        return _normalized_activation(raw, self.SPEC.dynamic_activation)

    def forward(
        self,
        data_dict: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dynamic_out = self._dynamic_head(data_dict["xc_nn_norm"])
        static_out = torch.sigmoid(self.ann(data_dict["c_nn_norm"]))
        return dynamic_out, static_out

    def predict_timevar_parameters(self, z1: torch.Tensor) -> torch.Tensor:
        dynamic_out = self._dynamic_head(z1)
        if self.nmul > 0 and self.ny1 % self.nmul == 0:
            return dynamic_out.reshape(
                dynamic_out.shape[0],
                dynamic_out.shape[1],
                self.dynamic_param_count,
                self.nmul,
            )
        return dynamic_out

    def predict_timevar_parametersv2(self, z1: torch.Tensor) -> torch.Tensor:
        return self._dynamic_head(z1)


class S4DBaseline(S5DAblationMlp):
    SPEC = AblationSpec(
        name="S4D-baseline",
        add_conv=False,
        norm="batch",
        dynamic_activation="sigmoid",
        purpose="original reference model",
    )


class S4DLN(S5DAblationMlp):
    SPEC = AblationSpec(
        name="S4D-LN",
        add_conv=False,
        norm="layer",
        dynamic_activation="sigmoid",
        purpose="isolate normalization change only",
    )


class S4DSoftsign(S5DAblationMlp):
    SPEC = AblationSpec(
        name="S4D-Softsign",
        add_conv=False,
        norm="batch",
        dynamic_activation="softsign",
        purpose="isolate activation change only",
    )


class S4DLNSoftsign(S5DAblationMlp):
    SPEC = AblationSpec(
        name="S4D-LN-Softsign",
        add_conv=False,
        norm="layer",
        dynamic_activation="softsign",
        purpose="test normalization and activation without convolution",
    )


class S5DConvOnly(S5DAblationMlp):
    SPEC = AblationSpec(
        name="S5D-ConvOnly",
        add_conv=True,
        norm="batch",
        dynamic_activation="sigmoid",
        purpose="isolate convolution under original normalization and activation",
    )


class S5DFull(S5DAblationMlp):
    SPEC = AblationSpec(
        name="S5D-full",
        add_conv=True,
        norm="layer",
        dynamic_activation="softsign",
        purpose="final proposed S5D configuration",
    )
