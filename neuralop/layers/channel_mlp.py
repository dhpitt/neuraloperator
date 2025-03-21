from typing import List, Literal

from torch import nn
import torch.nn.functional as F

# dispatch nonlinearity to avoid serializing nn.Functional modules
nonlinearity_modules = {'gelu': F.gelu,
                        'relu': F.relu,
                        'elu': F.elu,
                        'tanh': F.tanh,
                        'sigmoid': F.sigmoid}

class ChannelMLP(nn.Module):
    """ChannelMLP applies an arbitrary number of layers of 
    1d convolution and nonlinearity to the channels of input
    and is invariant to spatial resolution.

    Parameters
    ----------
    in_channels : int
        number of input channels
    out_channels : int, default is None
        number of output channels
        if None, same is in_channels
    hidden_channels : int, default is None
        number of hidden channels
        if None, same is in_channels
    n_layers : int, default is 2
        number of linear layers in the MLP
    non_linearity : Literal ["gelu", "relu", "elu", "sigmoid", "tanh"],
        Non-linear activation function to use, by default "gelu" (F.gelu)
    dropout : float, default is 0
        if > 0, dropout probability
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int=None,
        hidden_channels: int=None,
        n_layers: int=2,
        non_linearity: Literal['gelu', 'relu', 'elu', 'sigmoid', 'tanh']='gelu',
        dropout: float=0.0,
        **kwargs,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels = in_channels
        self.out_channels = in_channels if out_channels is None else out_channels
        self.hidden_channels = (
            in_channels if hidden_channels is None else hidden_channels
        )
        self.non_linearity = nonlinearity_modules[non_linearity]
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(n_layers)])
            if dropout > 0.0
            else None
        )
        
        # we use nn.Conv1d for everything and roll data along the 1st data dim
        self.fcs = nn.ModuleList()
        for i in range(n_layers):
            if i == 0 and i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.in_channels, self.out_channels, 1))
            elif i == 0:
                self.fcs.append(nn.Conv1d(self.in_channels, self.hidden_channels, 1))
            elif i == (n_layers - 1):
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.out_channels, 1))
            else:
                self.fcs.append(nn.Conv1d(self.hidden_channels, self.hidden_channels, 1))

    def forward(self, x):
        reshaped = False
        size = list(x.shape)
        if x.ndim > 3:  
            # batch, channels, x1, x2... extra dims
            # .reshape() is preferable but .view()
            # cannot be called on non-contiguous tensors
            x = x.reshape((*size[:2], -1)) 
            reshaped = True

        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        # if x was an N-d tensor reshaped into 1d, undo the reshaping
        # same logic as above: .reshape() handles contiguous tensors as well
        if reshaped:
            x = x.reshape((size[0], self.out_channels, *size[2:]))

        return x


# Reimplementation of the ChannelMLP class using Linear instead of Conv
class LinearChannelMLP(nn.Module):
    """LinearChannelMLP applies an arbitrary number of nn.Linear layers
    and nonlinearity to the channels of input and is invariant to spatial resolution.

    Parameters
    ----------
    layers: List[int]
        list of linear layer widths, so that
        ``self.layers[i] = nn.Linear(layers[i], layers[i+1])``
    non_linearity : Literal ["gelu", "relu", "elu", "sigmoid", "tanh"],
        Non-linear activation function to use, by default "gelu" (F.gelu)
    dropout : float, default is 0
        if > 0, dropout probability
    """
    def __init__(self, 
                 layers: List[int],
                 non_linearity: Literal['gelu', 'relu', 'elu', 'sigmoid', 'tanh']='gelu',
                 dropout: float=0.0):
        super().__init__()

        self.n_layers = len(layers) - 1

        assert self.n_layers >= 1

        self.fcs = nn.ModuleList()
        self.non_linearity = nonlinearity_modules[non_linearity]
        self.dropout = (
            nn.ModuleList([nn.Dropout(dropout) for _ in range(self.n_layers)])
            if dropout > 0.0
            else None
        )

        for j in range(self.n_layers):
            self.fcs.append(nn.Linear(layers[j], layers[j + 1]))

    def forward(self, x):
        for i, fc in enumerate(self.fcs):
            x = fc(x)
            if i < self.n_layers - 1:
                x = self.non_linearity(x)
            if self.dropout is not None:
                x = self.dropout[i](x)

        return x
