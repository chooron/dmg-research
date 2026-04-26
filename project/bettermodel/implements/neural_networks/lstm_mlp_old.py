from typing import Optional

import torch
import torch.nn.functional as F

from project.bettermodel.implements.neural_networks.layers.ann import AnnModel
from project.bettermodel.implements.neural_networks.layers.cudnn_lstm import CudnnLstmModel
from project.bettermodel.implements.neural_networks.layers.lstm import LstmModel


class LstmMlpModel(torch.nn.Module):
    """LSTM-MLP model for multi-scale learning.
    
    这个是MHPI自己的写的版本,执行了一些手动化操作
    
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
        device: Optional[str] = 'cpu',
    ) -> None:
        super().__init__()
        self.name = 'LstmMlpModel'

        if device == 'cpu':
            # CPU-compatible LSTM model.
            self.lstminv = LstmModel(
                nx=nx1, ny=ny1, hidden_size=hiddeninv1, dr=dr1,
            )
        else:
            # GPU-only HydroDL LSTM.
            self.lstminv = CudnnLstmModel(
                nx=nx1, ny=ny1, hidden_size=hiddeninv1, dr=dr1,
            )

        self.ann = AnnModel(
            nx=nx2, ny=ny2, hidden_size=hiddeninv2, dr=dr2,
        )

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Parameters
        ----------
        z1
            The LSTM input tensor.
        z2
            The MLP input tensor.
        
        Returns
        -------
        tuple
            The LSTM and MLP output tensors.
        """
        lstm_out = self.lstminv(z1)  # dim: timesteps, gages, params
        ann_out = self.ann(z2)
        return F.sigmoid(lstm_out), F.sigmoid(ann_out)
