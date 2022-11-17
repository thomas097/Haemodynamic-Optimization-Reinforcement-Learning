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
    def __init__(self, in_channels, out_channels, omega_0=1.0, use_bias=True):
        """ Implements a Sine layer with tunable omega_0 as in (Romero et al., 2021)
        For details: https://arxiv.org/pdf/2102.02611.pdf
        :param in_channels:   Number of input features
        :param out_channels:  Number of output features
        :param omega_0:       Initial guess of frequency parameter
        :param use_bias:      Whether to include an additive bias
        """
        super().__init__()
        self._linear = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=use_bias)
        self._omega_0 = torch.nn.Parameter(torch.tensor(omega_0))

    def forward(self, x):
        """ Forward pass through Sine layer
        :param x:  Input tensor of shape (batch_size, in_channels, n_timesteps)
        :returns:  Tensor of shape (batch_size, out_channels, n_timesteps)
        """
        return torch.sin(self._omega_0 * self._linear(x))


class LayerNorm(torch.nn.Module):
    def __init__(self, in_channels, eps=1e-12):
        """ Implementation of LayerNorm using GroupNorm """
        super().__init__()
        self.layer_norm = torch.nn.GroupNorm(1, num_channels=in_channels, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)


class KernelNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels, d_kernel, omega_0=1.0, use_bias=True):
        """ Implementation of a SIREN kernel network used to model a bank of
        convolutional filters. For details: https://arxiv.org/abs/2006.09661
        :param in_channels:   Number of input features
        :param out_channels:  Number of output features
        :param d_kernel:      Number of hidden units of kernel network
        :param omega_0:       Initial guess of frequency parameter
        :param use_bias:      Whether to include an additive bias
        """
        super().__init__()
        self.__kernel = torch.nn.Sequential(
            Sine(1, d_kernel, omega_0=omega_0, use_bias=use_bias),
            Sine(d_kernel, d_kernel, omega_0=omega_0, use_bias=use_bias),
            torch.nn.Conv1d(d_kernel, in_channels * out_channels, kernel_size=1)
        )
        self._initialize(omega_0)

    def _initialize(self, omega_0):
        """ Initializes weights of SIREN """
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

    def forward(self, x):
        """ Forward pass through kernel network
        :param x:  Tensor of relative positions of shape (1, 1, n_timesteps)
        :returns:  Kernel of shape (1, in_channels * out_channels, n_timesteps)
        """
        return self.__kernel(x)


class CKConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, max_timesteps, d_kernel=16, use_bias=True):
        """ CKConv layer
        :param in_channels:    Number of input features
        :param out_channels:   Number of output features
        :param max_timesteps:  Maximum number of timesteps in input
        :param d_kernel:       Dimensions of the hidden layers of the kernel network
        :param use_bias:       Whether to include an additive bias
        """
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        # kernel network + bias
        self._kernel = KernelNet(in_channels=in_channels,
                                 out_channels=out_channels,
                                 d_kernel=d_kernel)

        self._bias = torch.nn.Parameter(torch.randn(out_channels)) if use_bias else None

        # precompute relative positions
        max_timesteps = max_timesteps + 1 if max_timesteps % 2 == 0 else max_timesteps
        rel_positions = torch.linspace(-1, 1, max_timesteps).unsqueeze(0).unsqueeze(0)
        self.register_buffer('_rel_positions', rel_positions, persistent=True)

    @property
    def kernel(self):
        """ Returns ndarray representing kernel for visualization purposes """
        return self._kernel(self._rel_positions).view(self._out_channels, self._in_channels, -1).cpu().detach().numpy()

    def forward(self, x):
        """ Forward pass through CKConv layer
        :param x:  Input tensor of shape (batch_size, in_channels, n_timesteps)
        :returns:  Output tensor of shape (batch_size, out_channels, n_timesteps)
        """
        kernel = self._kernel(self._rel_positions).view(self._out_channels, self._in_channels, -1)
        signal = self._causal_padding(x, kernel)
        return F.conv1d(signal, kernel, bias=self._bias, padding=0)

    @staticmethod
    def _causal_padding(x, kernel):
        """ Performs left-padding of input sequence with kernel_size - 1 zeros
        :param x:       Input of shape (batch_size, in_channels, n_timesteps)
        :param kernel:  Weight matrix of shape (out_channels, in_channels, n_timesteps)
        :returns:       Input x padded to length |x| + kernel_size - 1
        """
        return F.pad(x, [kernel.shape[-1] - 1, 0], value=0.0)


class CKBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, max_timesteps, d_kernel, use_bias=True, use_residual=False):
        """ CKConv block with layer normalization and optional residual connections (Bai et al., 2017)
        :param in_channels:    Number of input features
        :param out_channels:   Number of output features
        :param max_timesteps:  Maximum number of timesteps in input
        :param d_kernel:       Dimensions of the hidden layers of the kernel network
        :param use_bias:       Whether to include an additive bias (default: True)
        :param use_residual:   Whether to include a residual connection inside the block (default: False)
        """
        super().__init__()
        # CKConv layer + Activation
        self.ckconv = CKConv(in_channels, out_channels, max_timesteps, d_kernel=d_kernel, use_bias=use_bias)
        self.layer_norm = LayerNorm(out_channels)
        self.leaky_relu = torch.nn.LeakyReLU()

        # for skip connections, apply Conv1D to input if in_channels != out_channels
        if use_residual:
            if in_channels != out_channels:
                self._res_conn = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
            else:
                self._res_conn = torch.nn.Identity()
        else:
            self._res_conn = None

    def forward(self, x):
        """ Forward pass through layer
        :param x:  Input tensor of shape (batch_size, in_channels, n_timesteps)
        :returns:  Output tensor of shape (batch_size, out_channels, n_timesteps)
        """
        out = self.ckconv(x)
        if self._res_conn is not None:
            out = out + self._res_conn(x)
        return self.leaky_relu(self.layer_norm(out))


if __name__ == '__main__':
    X = torch.randn((64, 16, 1000))

    # Create stack of CKConv layers
    conv = CKBlock(
        in_channels=16,
        out_channels=32,
        max_timesteps=1000,
        d_kernel=8
    )

    # Sanity check: time forward pass
    def timing_test(model, inputs, N=10):
        time1 = time.time_ns()
        for _ in range(N):
            y = model(inputs)
        time_elapsed = (time.time_ns() - time1) / (1e9 * N)
        return time_elapsed, y

    elapsed_time, output = timing_test(conv, X)
    print('Time elapsed:', elapsed_time)
    print('Out.shape:   ', output.shape)
