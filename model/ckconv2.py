"""
Author:   Thomas Bellucci
Filename: ckconv.py
Descr.:   Implements a 1D convolutional layer using a continuous kernel parameterization
          (Sitzmann et al., 2020) for irregularly-sampled time series.
          Code is based on an earlier implementation of CKConv (Romero et al., 2021),
          which can be found at https://github.com/dwromero/ckconv. Our version unrolls
          the convolution operation into a matrix-product, allowing the kernel to vary
          at each time step as a function of the other point's relative positions.
Date:     01-10-2022
"""

import time
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
    Implementation of a continuous SIREN kernel network used to
    continuously model a bank of convolutional filters.
    """
    def __init__(self, input_units, hidden_units, output_units, omega_0=32.5, use_bias=True):
        super().__init__()
        self.__kernel = torch.nn.Sequential(
            Sine(input_units, hidden_units, omega_0=omega_0, bias=use_bias),
            Sine(hidden_units, hidden_units, omega_0=omega_0, bias=use_bias),
            torch.nn.Linear(hidden_units, output_units)
        )

    def forward(self, t):
        return self.__kernel(t)


class CKConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_hidden_units=32):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        self._kernel = KernelNet(input_units=1,
                                 hidden_units=kernel_hidden_units,
                                 output_units=in_channels * out_channels)

        # Max-length seen during training
        self._train_max_len = torch.as_tensor(1.0)

        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

    def forward(self, x, t):
        # If training and current time span exceeds max span seen previously, update max length
        if self.training:
            self._train_max_len = torch.max(self._train_max_len, torch.max(t) - torch.min(t))

        # Create time-series with zeros in places of missing values (indicated by mask)
        x_new, t_new, mask = self._fill_missing_observations(x, t)

        # Sample weights from kernel network
        weights = self._kernel(t_new.unsqueeze(1))  # -> (seq_length, in_channels * out_channels)

        # Perform causal convolution on `x_new`
        z = self._causal_conv1d(x_new, weights)[0].T  # -> length first, channels last

        # TODO: normalize convolution output

        # Drop missing entries
        return z[mask, :]

    def _causal_conv1d(self, x, weights):
        # -> (batch_size, in_channels, seq_length)
        inputs = x.permute(0, 2, 1)

        # -> (out_channels, in_channels, seq_length)
        kernel = weights.T.view(self._out_channels, self._in_channels, *inputs.shape[2:])

        x, kernel = self._causal_padding(inputs, kernel)
        return F.conv1d(x, kernel, padding=0)

    @staticmethod
    def _causal_padding(x, kernel):
        if kernel.shape[-1] % 2 == 0:
            kernel = F.pad(kernel, [1, 0], value=0.0)
        x = F.pad(x, [kernel.shape[-1] - 1, 0], value=0.0)
        return x, kernel

    def _fill_missing_observations(self, x, t):
        # Compute length (N) of series `x` if it were complete
        t_min, t_max = torch.min(t), torch.max(t)
        N = t_max - t_min + 1

        # Compute non-unitary timesteps in range [-1, 1] (will exceed +1 if `N > train_max_len`)
        t_new = torch.linspace(-1, 2 * (N / self._train_max_len) - 1, N)

        # Create sequence with `x` and missing entries as zeros
        x_new = torch.zeros((1, N) + x.shape[1:])
        x_new[0, t - t_min] = x

        # Create mask indicating missing values
        mask = torch.zeros(N, dtype=torch.bool)
        mask[t - t_min] = 1
        return x_new, t_new, mask


if __name__ == '__main__':
    # Create toy dataset of N points of one measurement type (e.g. HeartRate)
    N = 1000
    D_in, D_out = 1, 16
    t = torch.cumsum(torch.randint(1, 3, (N,)), dim=0)
    x = torch.abs(torch.randn((N, D_in)))
    print('In: x =', x.shape, ' t =', t.shape)

    # Create stack of CKConv layers
    layer1 = CKConv(in_channels=D_in, out_channels=D_out)
    layer1.train(True)

    # Time operation
    N_repeats = 1000
    time1 = time.time_ns()
    for _ in range(N_repeats):
        out = layer1(x, t)

    print('Time elapsed:', (time.time_ns() - time1) / (1e9 * N_repeats), '\n')
    print('Out:', out.shape)
