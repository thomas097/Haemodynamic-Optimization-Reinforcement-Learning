"""
Author:   Thomas Bellucci
Filename: ckconv1d.py
Descr.:   Implements a 1D-conv layer with a continuous SIREN parameterization
          (Sitzmann et al., 2020) for irregularly-sampled time series.
          Code is based on an earlier implementation of CKConv (Romero et al., 2021),
          which can be found at https://github.com/dwromero/ckconv. Note that our
          version unrolls the convolution to a matrix-product, allowing the kernel to
          vary slightly at each time step as a function of relative positions.
Date:     22-09-2022
"""

import time
import torch
from torch.nn.utils import weight_norm


class Sine(torch.nn.Module):
    """
    Implements the Sine layer as used by CKConv; sin(omega_0*[Wx+b]).
    For details: https://arxiv.org/pdf/2102.02611.pdf
    """
    def __init__(self, input_dims, output_dims, omega_0=1.0, bias=True):
        super().__init__()
        # Replace BatchNorm by weight_norm due to noise-sensitivity of DRL
        self._linear = weight_norm(torch.nn.Linear(input_dims, output_dims, bias=bias))
        self._omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self._omega_0 * self._linear(x))


class KernelNet(torch.nn.Module):
    """
    Implementation of a SIREN-based kernel network used to continuously
    represent a bank of convolutional kernels.
    """
    def __init__(self, input_units=3, hidden_units=32, omega_0=1.0, use_bias=True):
        super().__init__()
        # i.e. timestep + in_channel + out_channel -> kernel value
        self.__kernel = torch.nn.Sequential(
            Sine(input_units, hidden_units, omega_0=omega_0, bias=use_bias),
            Sine(hidden_units, hidden_units, omega_0=omega_0, bias=use_bias),
            torch.nn.Linear(hidden_units, 1),
        )

    def forward(self, t):
        return self.__kernel(t)


class CKConv1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, is_causal=True, kernelnet_units=32, use_bias=True):
        super().__init__()
        self._kernel = KernelNet(hidden_units=kernelnet_units,
                                 use_bias=use_bias)
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._is_causal = is_causal

    def forward(self, x, t):
        # Compute matrix of relative positions
        dt = self._handle_relative_positions(t)

        # Sample kernel at relative positions
        kernel = self._get_kernel_as_toeplitz(dt)

        # Mask kernel if causal convolution
        if self._is_causal:
            kernel = self._causal_mask(dt) * kernel

        # TODO: inverse density to normalize kernel (see paper)

        # Flatten input column-wise
        x_input = x.t().flatten()

        # Convolution!
        return torch.matmul(kernel, x_input).reshape(self._out_channels, x.shape[0]).t()

    def _causal_mask(self, dt):
        # Mask upper-triangle of kernel to ensure causal convolution
        mask = 1 - torch.triu(torch.ones(dt.shape), diagonal=1)
        return mask.repeat(self._out_channels, self._in_channels)

    def _get_kernel_as_toeplitz(self, dt):
        # Repeat Toeplitz matrix to incorporate more in_channels and out_channels
        deltas = dt.repeat(self._out_channels, self._in_channels)

        # For each entry, determine in_channel and out_channel (kernel) indices
        # TODO: Simplify (ultra slow rn)
        in_channels = torch.arange(self._in_channels)[None].repeat_interleave(dt.shape[0], dim=1).repeat(dt.shape[0] * self._out_channels, 1)
        out_channels = torch.arange(self._out_channels)[:, None].repeat_interleave(dt.shape[0], dim=0).repeat(1, dt.shape[0] * self._in_channels)

        # Concatenate flattened matrices to (N * N * in_channels, 2) matrix
        x_kernel = torch.concat([deltas.reshape(-1, 1),
                                 in_channels.reshape(-1, 1),
                                 out_channels.reshape(-1, 1)], dim=1)

        # Sample kernel and reshape to (|x|, |x| * in_channels)
        return self._kernel(x_kernel).reshape(*deltas.shape)

    @staticmethod
    def _handle_relative_positions(t):
        # Compute matrix of relative positions
        t_mat = torch.unsqueeze(t, 0).repeat(t.shape[0], 1)
        return t_mat - t_mat.t()


if __name__ == '__main__':
    x = torch.randn(80, 1)
    t = torch.randn(80)

    conv1 = CKConv1D(in_channels=1, out_channels=5)
    conv2 = CKConv1D(in_channels=5, out_channels=8)

    time1 = time.time_ns()
    y = conv1(x, t)
    z = conv2(y, t)
    print('Time elapsed:', (time.time_ns() - time1) / 1e9)
