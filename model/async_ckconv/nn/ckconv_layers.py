import time
import numpy as np
import torch
import torch.nn.functional as F


class Sine(torch.nn.Module):
    def __init__(self, in_channels, out_channels, omega_0=1.0, use_bias=True):
        """
        Implements a Sine layer with tunable omega_0 as used in (Romero et al., 2021)
        For details, see: https://arxiv.org/pdf/2102.02611.pdf
        :param in_channels:   Number of input channels / features
        :param out_channels:  Number of output channels / features
        :param omega_0:       Initial estimate of frequency parameter
        :param use_bias:      Whether to include an additive bias
        """
        super().__init__()
        self._linear = torch.nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=use_bias)
        self._omega_0 = torch.nn.Parameter(torch.tensor(omega_0))

    def forward(self, x):  # <- (batch_size, num_channels, seq_length)
        return torch.sin(self._linear(x))


class KernelNet(torch.nn.Module):
    def __init__(self, hidden_channels, out_channels, omega_0=1.0, use_bias=True):
        """
        Implementation of a kernel network using SIRENs (Sitzmann et al., 2020)
        to model a bank of convolutional filters.
        For details, see https://arxiv.org/abs/2006.09661
        :param hidden_channels:  Number of input channels / features
        :param out_channels:     Number of output channels / features
        :param omega_0:          Initial estimate of frequency parameter
        :param use_bias:         Whether to include an additive bias
        """
        super().__init__()
        self.__kernel = torch.nn.Sequential(
            Sine(1, hidden_channels, omega_0=omega_0, use_bias=use_bias),
            Sine(hidden_channels, hidden_channels, omega_0=omega_0, use_bias=use_bias),
            torch.nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        )
        self._initialize(omega_0)

    def _initialize(self, omega_0):
        """
        Initializes the weights of the SIREN
        :param omega_0: Initial guess of frequency parameter
        """
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
        """
        Forward pass through SIREN
        :param x:  Kernel input of relative positions of shape (1, 1, |x|)
        :returns:  Tensor of weights of shape (1, out_channels, |x|)
        """
        return self.__kernel(x)


class CKConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, max_timesteps, d_kernel=16, use_bias=True):
        """
        Continuous Kernel Convolutional layer (Romero et al., 2021)
        :param in_channels:    Number of input channels / features
        :param out_channels:   Number of output channels / features
        :param max_timesteps:  Max number of positions in input
        :param d_kernel:       Latent dimensionality of kernel network
        :param omega_0:        Initial estimate of frequency parameter
        :param use_bias:       Whether to include an additive bias
        """
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels

        # kernel network
        self._net = KernelNet(out_channels=in_channels * out_channels,
                              hidden_channels=d_kernel)
        # learn bias directly
        self._bias = torch.nn.Parameter(torch.randn(out_channels)) if use_bias else None

        # precompute relative positions vector
        pos_abs = max_timesteps + 1 if max_timesteps % 2 == 0 else max_timesteps
        pos_rel = torch.linspace(-1, 1, pos_abs).unsqueeze(0).unsqueeze(0)
        self.register_buffer('_pos_rel', pos_rel, persistent=True)

    @staticmethod
    def _causal_padding(inputs, kernel):
        """ Performs left-padding of input sequence with kernel_size - 1 zeros
        :param inputs:  Input signal
        :param kernel:  Kernel with which to convolve the input
        :returns:       Signal padded with zeros
        """
        return F.pad(inputs, pad=[kernel.shape[-1] - 1, 0], value=0.0)

    def forward(self, x, return_kernels=False):
        """
        Forward pass through CKConv
        :param x:              Input signal
        :param return_kernels: Whether to include the sampeled kernels in the output (default: False)
        :returns:              Signal convolved with kernel sampled from kernel net
        """
        kernel = self._net(self._pos_rel).view(self._out_channels, self._in_channels, -1)

        inputs = self._causal_padding(x, kernel=kernel)
        out = F.conv1d(inputs, weight=kernel, bias=self._bias, padding=0)

        return (out, kernel) if return_kernels else out
