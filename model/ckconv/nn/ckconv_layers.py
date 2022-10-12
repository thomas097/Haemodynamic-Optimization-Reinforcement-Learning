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
        self._linear = weight_norm(torch.nn.Conv1d(input_units, output_units, kernel_size=1, bias=bias))
        self._omega_0 = omega_0

    def forward(self, x):
        # x -> [batch_size, num_channels, seq_length]
        return torch.sin(self._omega_0 * self._linear(x))


class KernelNet(torch.nn.Module):
    """
    Implementation of a SIREN kernel network used to continuously model
    a bank of convolutional filters.
    For details: https://arxiv.org/abs/2006.09661
    """
    def __init__(self, hidden_units, output_units, omega_0=32.5, use_bias=True):
        super().__init__()
        self.__kernel = torch.nn.Sequential(
            Sine(1, hidden_units, omega_0=omega_0, bias=use_bias),
            Sine(hidden_units, hidden_units, omega_0=omega_0, bias=use_bias),
            weight_norm(torch.nn.Conv1d(hidden_units, output_units, kernel_size=1))
        )
        self.initialize(omega_0)

    def initialize(self, omega_0):
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

    def forward(self, t):
        return self.__kernel(t)


class CKConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_dims=32, use_bias=True):
        super().__init__()
        # Kernel network
        self._kernel = KernelNet(output_units=in_channels * out_channels,
                                 hidden_units=kernel_dims)

        self._bias = torch.nn.Parameter(torch.zeros(out_channels)) if use_bias else None

        # Max sequence length seen during training
        self._train_max_len = torch.as_tensor(-1.0)
        self._rel_positions = None
        self._in_channels = in_channels
        self._out_channels = out_channels

        # Use GPU when available
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

    def forward(self, x):
        # Update max_len when length of training example exceeds max_len seen previously
        if self.training and x.shape[1] > self._train_max_len.item():
            self._train_max_len = torch.as_tensor(x.shape[1])
            self._rel_positions = torch.linspace(-1, 1, x.shape[1]).unsqueeze(0).unsqueeze(0)
            # -> (batch_size=1, num_channels=1, seq_length)

        weights = self._kernel(self._rel_positions).view(self._out_channels, self._in_channels, -1)

        out = self._causal_conv1d(x, weights).permute(0, 2, 1)
        # -> (batch_size, num_timesteps, out_channels)

        # TODO: normalize to de-bias convolution output?
        return out

    def _causal_conv1d(self, inputs, weights):
        inputs = inputs.permute(0, 2, 1)
        inputs, kernel = self._causal_padding(inputs, weights)
        return F.conv1d(inputs, kernel, bias=self._bias, padding=0)

    @staticmethod
    def _causal_padding(x, kernel):
        # Pad even-length kernel
        if kernel.shape[-1] % 2 == 0:
            kernel = F.pad(kernel, [1, 0], value=0.0)

        # Left-pad input with zeros
        x = F.pad(x, [kernel.shape[-1] - 1, 0], value=0.0)
        return x, kernel


class CKBlock(torch.nn.Module):
    """ CKConv Block with layer normalization and residual connections
        (Bai et. al., 2017)
    """
    def __init__(self, num_channels, hidden_channels=None, kernel_dims=32, use_bias=True):
        super().__init__()
        # Use hidden == out if hidden_channels are not provided
        hidden_channels = num_channels if hidden_channels is None else hidden_channels

        self._ckconv1 = CKConv(num_channels, hidden_channels, kernel_dims, use_bias=use_bias)
        self._ckconv2 = CKConv(hidden_channels, num_channels, kernel_dims, use_bias=use_bias)
        self._layer_norm1 = torch.nn.LayerNorm(hidden_channels)
        self._layer_norm2 = torch.nn.LayerNorm(num_channels)
        self._leaky_relu = torch.nn.LeakyReLU()

        # Use GPU when available
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

    def forward(self, x):
        h1 = self._leaky_relu(self._layer_norm1(self._ckconv1(x)))
        h2 = self._leaky_relu(self._layer_norm2(self._ckconv2(h1)))
        return self._leaky_relu(x + h2)


if __name__ == '__main__':
    # Create toy dataset of shape x=(batch_size, seq_len, in_channels)
    # with ~20% observations missing (NaNs)
    x = torch.randn((64, 1000, 16))
    mask = torch.rand((64, 1000, 16)) > 0.2
    dataset = x * mask
    print('In: ', dataset.shape)

    # Create stack of CKConv layers
    conv = CKBlock(num_channels=16, hidden_channels=32)
    conv.train(True)

    # Sanity check: time forward pass
    def timing_test(model, inputs, N=10):
        time1 = time.time_ns()
        for _ in range(N):
            model(inputs)
        time_elapsed = (time.time_ns() - time1) / (1e9 * N)
        return time_elapsed

    print('Time elapsed:', timing_test(conv, dataset))
