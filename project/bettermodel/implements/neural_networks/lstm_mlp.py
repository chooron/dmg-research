from typing import Optional

import torch

from project.bettermodel.implements.neural_networks.layers.ann import AnnModel
from project.bettermodel.implements.neural_networks.layers.cudnn_lstm import CudnnLstmModel


class LstmMlpModel(torch.nn.Module):
    """LSTM-MLP model for multi-scale learning.

    Supports GPU and CPU forwarding.

    Parameters
    ----------
    nx1
        Number of LSTM input features.
    ny1
        Number of LSTM output features.
    hiddeninv1
        LSTM hidden size.
    nx2
        Number of MLP input features.
    ny2
        Number of MLP output features.
    hiddeninv2
        MLP hidden size.
    dr1
        Dropout rate for LSTM. Default is 0.5.
    dr2
        Dropout rate for MLP. Default is 0.5.
    cache_states
        Whether to cache hidden and cell states for LSTM.
    device
        Device to run the model on. Default is 'cpu'.
    """

    def __init__(
        self,
        *,
        nx1: int,
        ny1: int,
        hiddeninv1: int,
        nx2: int,
        ny2: int,
        hiddeninv2: int,
        dr1: Optional[float] = 0.5,
        dr2: Optional[float] = 0.5,
        cache_states: Optional[bool] = False,
        device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'LstmMlpModel'
        self.nx1 = nx1
        self.ny1 = ny1
        self.hiddeninv1 = hiddeninv1
        self.nx2 = nx2
        self.ny2 = ny2
        self.hiddeninv2 = hiddeninv2
        self.dr1 = dr1
        self.dr2 = dr2
        self.cache_states = cache_states
        self.device = device

        self.hn, self._hn_cache = None, None  # hidden state
        self.cn, self._cn_cache = None, None  # cell state

        self.lstminv = CudnnLstmModel(
            nx=nx1,
            ny=ny1,
            hidden_size=hiddeninv1,
            dr=dr1,
            cache_states=cache_states,
        )
        self._lstm_output_activated = False

        self.activation = torch.nn.Sigmoid()

        self.ann = AnnModel(
            nx=nx2,
            ny=ny2,
            hidden_size=hiddeninv2,
            dr=dr2,
        )

    @classmethod
    def build_by_config(cls, config: dict, device: Optional[str] = 'cpu'):
        return cls(
            nx1=config["nx1"] if "nx1" in config else config["nx"],
            ny1=config["ny1"],
            hiddeninv1=config["lstm_hidden_size"],
            nx2=config["nx2"],
            ny2=config["ny2"],
            hiddeninv2=config["mlp_hidden_size"],
            dr1=config["lstm_dropout"],
            dr2=config["mlp_dropout"],
            cache_states=config.get("cache_states", False),
            device=device,
        )

    def get_states(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get hidden and cell states."""
        return self._hn_cache, self._cn_cache

    def load_states(
        self,
        states: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Load hidden and cell states."""
        if not (isinstance(states, tuple) and len(states) == 2):
            raise ValueError("`states` must be a tuple of 2 tensors.")
        for state in states:
            if state is not None and not isinstance(state, torch.Tensor):
                raise ValueError("Each element in `states` must be a tensor.")

        if states[0] is None or states[1] is None:
            self.hn, self.cn = None, None
            return

        device = next(self.parameters()).device
        self.hn = states[0].detach().to(device)
        self.cn = states[1].detach().to(device)
        if hasattr(self.lstminv, "load_states"):
            self.lstminv.load_states((self.hn, self.cn))

    def _forward_lstm(self, x1: torch.Tensor) -> torch.Tensor:
        lstm_out = self.lstminv(x1)
        if hasattr(self.lstminv, "get_states"):
            self._hn_cache, self._cn_cache = self.lstminv.get_states()

        if self.cache_states and self._hn_cache is not None and self._cn_cache is not None:
            self.hn = self._hn_cache.to(x1.device)
            self.cn = self._cn_cache.to(x1.device)

        if self._lstm_output_activated:
            return lstm_out
        return self.activation(lstm_out)

    def forward(
        self,
        x1: dict | torch.Tensor,
        x2: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        NOTE (caching): Hidden states are always cached so that they can be
        accessed by `get_states`, but they are only available to the LSTM if
        `cache_states` is set to True.

        Parameters
        ----------
        x1
            The LSTM input tensor.
        x2
            The MLP input tensor.

        Returns
        -------
        tuple
            The LSTM and MLP output tensors.
        """
        if isinstance(x1, dict):
            data_dict = x1
            x1 = data_dict["xc_nn_norm"]
            x2 = data_dict["c_nn_norm"]
        if x2 is None:
            raise ValueError("x2 is required when forward input is not a data dict.")

        act_out = self._forward_lstm(x1)
        ann_out = self.activation(self.ann(x2))

        return (act_out, ann_out)

    def predict_timevar_parameters(self, z1: torch.Tensor) -> torch.Tensor:
        dynamic_out = self._forward_lstm(z1)
        return dynamic_out.reshape(dynamic_out.shape[0], 3, -1)

    def predict_timevar_parametersv2(self, z1: torch.Tensor) -> torch.Tensor:
        dynamic_out = self._forward_lstm(z1.permute(1, 0, 2))
        return dynamic_out.permute(1, 0, 2)
