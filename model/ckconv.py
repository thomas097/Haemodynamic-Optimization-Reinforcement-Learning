"""
Author:   Thomas Bellucci
Filename: ckconv.py
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
    def __init__(self, input_units, output_units, omega_0=1.0, bias=True):
        super().__init__()
        # Replace BatchNorm by weight_norm due to noise-sensitivity of DRL
        self._linear = weight_norm(torch.nn.Linear(input_units, output_units, bias=bias))
        self._omega_0 = omega_0

    def forward(self, x):
        return torch.sin(self._omega_0 * self._linear(x))


class KernelNet(torch.nn.Module):
    """
    Implementation of a SIREN-based kernel network used to continuously
    represent a bank of convolutional kernels.
    """
    def __init__(self, input_units, hidden_units, output_units, omega_0=32.5, use_bias=True):
        super().__init__()
        self.__kernel = torch.nn.Sequential(
            Sine(input_units, hidden_units, omega_0=omega_0, bias=use_bias),
            Sine(hidden_units, hidden_units, omega_0=omega_0, bias=use_bias),
            torch.nn.Linear(hidden_units, output_units),
        )

    def forward(self, t):
        return self.__kernel(t)


class CKConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_hidden_units=32, is_causal=True):
        super().__init__()
        # Kernel is indexed by timestep
        self._kernel = KernelNet(input_units=1,
                                 hidden_units=kernel_hidden_units,
                                 output_units=in_channels * out_channels)
        # Set GPU if available
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._is_causal = is_causal

    def forward(self, x, t, t_eval):
        # Compute matrix with rel. positions of points in `t` w.r.t. points in `t_eval`
        rel_mat = self._handle_relative_positions(t, t_eval)

        # Sample kernel for all deltas in rel_mat
        kernel = self._kernel(rel_mat.reshape(-1, 1))

        # Convert (|t| * |t_eval|, in_channels * out_channels) kernel to Toeplitz form
        toeplitz_kernel = self._kernel_to_toeplitz(kernel, t, t_eval)

        # Mask kernel if `is_causal` is enabled
        if self._is_causal:
            toeplitz_kernel = toeplitz_kernel * self._causal_mask(rel_mat)

        # Perform convolution of values with time-varying kernel as matmul
        x_flat = x.T.flatten()
        return torch.matmul(toeplitz_kernel, x_flat).reshape(self._out_channels, -1).T

    def _kernel_to_toeplitz(self, kernel, t, t_eval):
        # UNCOMMENT FOR DEBUGGING
        # kernel = torch.arange(len(t_eval) * len(t) * self._in_channels * self._out_channels)
        # kernel = kernel.reshape(len(t_eval) * len(t), self._in_channels * self._out_channels)

        # Cut `in_channels * out_channels` blocks out of kernel matrix to fill Toeplitz matrix
        interleaved_toeplitz = kernel.T.reshape(len(t_eval) * self._out_channels * self._in_channels, len(t))
        blocks = torch.tensor_split(interleaved_toeplitz, self._in_channels * self._out_channels, dim=0)

        # Concatenate blocks into `(|t_eval| * out_channels, |t| * in_channels)` Toeplitz
        return torch.concat(torch.tensor_split(torch.concat(blocks, dim=1), self._out_channels, dim=1), dim=0)

    @staticmethod
    def _handle_relative_positions(t_seq, t_eval):
        # Compute matrix of the deltas relative to `t_eval`
        mat0 = torch.unsqueeze(t_seq, 0).repeat(t_eval.shape[0], 1)
        mat1 = torch.unsqueeze(t_eval, 1).repeat(1, mat0.shape[1])
        return mat0 - mat1

    def _causal_mask(self, rel_mat):
        mask = (rel_mat <= 0.0).float()
        return mask.repeat(self._out_channels, self._in_channels)


if __name__ == '__main__':
    # Create toy dataset of N points of one measurement type (e.g. HeartRate)
    N = 200  # a sample approx. every hour for 72 hours
    t = torch.cumsum(torch.rand(N), dim=0)
    x = torch.abs(torch.randn((N, 16)))

    # Evaluate at each discrete point in t
    t_eval = t

    # Create stack of CKConv layers
    layer1 = CKConv(in_channels=16, out_channels=8)

    # Time operation
    time1 = time.time_ns()
    out = layer1(x, t, t_eval)
    print('Time elapsed:', (time.time_ns() - time1) / 1e9, '\n')

    print('Out:', out.shape)
