import torch
from ckconv_layers import CKConv


class CKCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._conv1 = CKConv(in_channels, out_channels)
        self._conv2 = CKConv(out_channels, out_channels)
        self._activation = torch.nn.ELU()

    def forward(self, x):
        h = self._activation(self._conv1(x))
        y = self._activation(self._conv2(h)) + h
        return y[:, -1]  # Return representation of final states