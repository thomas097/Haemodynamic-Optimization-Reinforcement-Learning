"""
Author:   Thomas Bellucci
Filename: ckconv.py
Descr.:   Implements a 1D convolutional layer using a continuous kernel parameterization
          using SIRENs (Sitzmann et al., 2020) meant for irregularly-sampled time series.
          Code is based on an earlier implementation of CKConv (Romero et al., 2021),
          which can be found at https://github.com/dwromero/ckconv. Our version unrolls
          the convolution operation into a matrix-product, allowing the kernel to vary
          at each time step as a function of the other point's relative positions.
Date:     01-10-2022
"""

import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm


class Sine(torch.nn.Module):
    """
    Implements the Sine layer used by CKConv (Romero et al., 2021).
    For details: https://arxiv.org/pdf/2102.02611.pdf
    """
    def __init__(self, input_units, output_units, omega_0=1.0, bias=True):
        super().__init__()
        # Replace BatchNorm by weight_norm due to noise-sensitivity of DRL
        self._linear = weight_norm(torch.nn.Linear(input_units, output_units, bias=bias))
        self._omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self._omega_0 * self._linear(x))


class KernelNet(torch.nn.Module):
    """
    Implementation of a SIREN kernel network used to continuously model
    a bank of convolutional filters.
    """
    def __init__(self, input_units, hidden_units, output_units, omega_0=32.5, use_bias=True):
        super().__init__()
        self.__kernel = torch.nn.Sequential(
            Sine(input_units, hidden_units, omega_0=omega_0, bias=use_bias),
            Sine(hidden_units, hidden_units, omega_0=omega_0, bias=use_bias),
            torch.nn.Linear(hidden_units, output_units)
        )
        self.initialize(omega_0)

    def initialize(self, omega_0):
        # initialization of SIREN (Please refer to https://arxiv.org/abs/2006.09661)
        first_layer = True
        for (i, layer) in enumerate(self.modules()):
            if isinstance(layer, torch.nn.Linear):
                # init uniformly if first layer
                if first_layer:
                    layer.weight.data.uniform_(-1, 1)
                    first_layer = False
                else:
                    init_val = np.sqrt(6.0 / layer.weight.shape[1]) / omega_0
                    layer.weight.data.uniform_(-init_val, init_val)

                # add bias to SIREN
                if layer.bias is not None:
                    layer.bias.data.uniform_(-1.0, 1.0)

    def forward(self, t):
        return self.__kernel(t)


class CKConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_hidden_units=32, use_bias=True):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self._kernel = KernelNet(input_units=1,
                                 hidden_units=kernel_hidden_units,
                                 output_units=in_channels * out_channels)

        if use_bias:
            self._bias = torch.nn.Parameter(torch.randn(out_channels))
            self._bias.requires_grad = True
        else:
            self._bias = None

        # Max-length seen during training
        self._train_max_len = torch.as_tensor(1.0)

        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

    def forward(self, x, mask=None):
        # When training and current time span exceeds max span seen previously, update max length
        if self.training:
            num_timesteps = torch.as_tensor(x.shape[1])
            self._train_max_len = torch.max(self._train_max_len, num_timesteps)

        # Compute non-unitary timesteps in range [-1, 1] (will exceed +1 if `N > train_max_len`)
        t_max = 2 * (x.shape[1] / self._train_max_len) - 1
        t = torch.linspace(-1, t_max, x.shape[1]).unsqueeze(1)

        # Sample kernel weights
        weights = self._kernel(t).T.view(self._out_channels, self._in_channels, -1)

        # Perform causal convolution with sampled convolution
        out = self._causal_conv1d(x, weights)

        # TODO: normalize to de-bias convolution output

        return out.permute(0, 2, 1)  # output.shape -> (batch_size, num_timesteps, out_channels)

    def _causal_conv1d(self, x, weights):
        # -> (batch_size, in_channels, seq_length)
        inputs = x.permute(0, 2, 1)

        x, kernel = self._causal_padding(inputs, weights)
        return F.conv1d(x, kernel, bias=self._bias, padding=0)

    @staticmethod
    def _causal_padding(x, kernel):
        if kernel.shape[-1] % 2 == 0:
            kernel = F.pad(kernel, [1, 0], value=0.0)
        x = F.pad(x, [kernel.shape[-1] - 1, 0], value=0.0)
        return x, kernel


if __name__ == '__main__':
    seq_length = 1000
    in_channels = 8
    out_channels = 16
    batch_size = 64

    # Create toy dataset of shape x=(batch_size, seq_len, in_channels)
    # with ~20% observations missing (NaNs)
    x = torch.abs(torch.randn((batch_size, seq_length, in_channels)))
    mask = torch.rand((batch_size, seq_length, in_channels)) > 0.5
    x = x * mask
    print('In: x =', x.shape)

    # Create stack of CKConv layers
    conv = CKConv(in_channels=in_channels, out_channels=out_channels)
    conv.train(True)

    # Time operation
    time1 = time.time_ns()
    out = conv(x, mask)
    time_elapsed = (time.time_ns() - time1) / 1e9

    print('Time elapsed:', time_elapsed, '\n')
    print('Out:', out.shape)
