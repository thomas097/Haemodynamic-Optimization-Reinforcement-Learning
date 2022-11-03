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


class Sine(torch.nn.Module):
    """
    Implements a Sine layer with tunable omega_0 (Romero et al., 2021)
    For details: https://arxiv.org/pdf/2102.02611.pdf
    """
    def __init__(self, input_units, output_units, omega_0=1.0, bias=True):
        super().__init__()
        self._linear = torch.nn.Conv1d(input_units, output_units, kernel_size=1, bias=bias)
        self._omega_0 = torch.nn.Parameter(torch.tensor(omega_0))

    def forward(self, x):  # <- (batch_size, num_channels, seq_length)
        return torch.sin(self._omega_0 * self._linear(x))


class LayerNorm(torch.nn.Module):
    """ Implementation of LayerNorm using ..GroupNorm """
    def __init__(self, in_channels, eps=1e-12):
        super().__init__()
        self.layer_norm = torch.nn.GroupNorm(1, num_channels=in_channels, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)


class KernelNet(torch.nn.Module):
    """
    Implementation of a SIREN kernel network used to continuously model
    a bank of convolutional filters.
    For details: https://arxiv.org/abs/2006.09661
    """
    def __init__(self, hidden_units, output_units, omega_0=1.0, use_bias=True):
        super().__init__()
        self.__kernel = torch.nn.Sequential(
            Sine(1, hidden_units, omega_0=omega_0, bias=use_bias),
            Sine(hidden_units, hidden_units, omega_0=omega_0, bias=use_bias),
            torch.nn.Conv1d(hidden_units, output_units, kernel_size=1)
        )
        self._initialize(omega_0)

    def _initialize(self, omega_0):
        # Initialize SIREN weights
        first_layer = True
        for (i, layer) in enumerate(self.modules()):
            if isinstance(layer, torch.nn.Conv1d):
                # init uniformly if first layer
                if first_layer:
                    layer.weight.data.uniform_(-1, 1)
                    first_layer = False
                else:
                    val = np.sqrt(6.0 / layer.weight.shape[1]) / omega_0
                    layer.weight.data.uniform_(-val, val)

    def forward(self, t):  # <- (1, 1, seq_length)
        return self.__kernel(t)


class CKConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, max_timesteps, kernel_dims=16, use_bias=True):
        super().__init__()
        # Kernel network + bias
        self._kernel = KernelNet(output_units=in_channels * out_channels,
                                 hidden_units=kernel_dims)

        self._bias = torch.nn.Parameter(torch.zeros(out_channels)) if use_bias else None

        # Use GPU when available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

        # Precompute relative positions (as we need target network with train=False)
        max_timesteps = max_timesteps + 1 if max_timesteps % 2 == 0 else max_timesteps
        self._rel_positions = torch.linspace(-1, 1, max_timesteps).unsqueeze(0).unsqueeze(0).to(self._device)
        self._in_channels = in_channels
        self._out_channels = out_channels

    @property
    def kernel(self):
        """ Returns ndarray representing kernel for visualization purposes """
        return self._kernel(self._rel_positions).view(self._out_channels, self._in_channels, -1).cpu().detach().numpy()

    def forward(self, x):  # <- (batch_size, num_timesteps, out_channels)
        kernel = self._kernel(self._rel_positions).view(self._out_channels, self._in_channels, -1)
        signal = self._causal_padding(x, kernel)
        return F.conv1d(signal, kernel, bias=self._bias, padding=0)

    @staticmethod
    def _causal_padding(x, kernel):
        # Left-pad input with zeros
        return F.pad(x, [kernel.shape[-1] - 1, 0], value=0.0)


class CKBlock(torch.nn.Module):
    """ Simplified CKConv Block with layer normalization and optional
        residual connections (Bai et. al., 2017)
    """
    def __init__(self, in_channels, out_channels, kernel_dims=32, max_timesteps=100, use_bias=True):
        super().__init__()
        # CKConv layer + Activation
        self.ckconv = CKConv(in_channels, out_channels, max_timesteps, kernel_dims, use_bias=use_bias)
        self.layer_norm = LayerNorm(out_channels)
        self.leaky_relu = torch.nn.LeakyReLU()

        # For skip connections, apply Conv1D to input if in_channels = out_channels
        if in_channels != out_channels:
            self._residual_conn = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self._residual_conn = torch.nn.Identity()

        # Use GPU when available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, x):  # <- (batch_size, in_channels, seq_length)
        return self.leaky_relu(self.layer_norm(self.ckconv(x)) + self._residual_conn(x))


if __name__ == '__main__':
    X = torch.randn((64, 1000, 16))

    # Create stack of CKConv layers
    conv = CKBlock(in_channels=16, out_channels=32)

    # Sanity check: time forward pass
    def timing_test(model, inputs, N=10):
        time1 = time.time_ns()
        for _ in range(N):
            model(inputs)
        time_elapsed = (time.time_ns() - time1) / (1e9 * N)
        return time_elapsed

    print('Time elapsed:', timing_test(conv, X))
