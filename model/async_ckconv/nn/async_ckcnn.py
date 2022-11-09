import torch
import torch.nn.functional as F
from async_ckconv import AsyncCKConv
from ckconv_layers import CKConv


class LayerNorm(torch.nn.Module):
    def __init__(self, in_channels, eps=1e-12):
        """ Implementation of LayerNorm using GroupNorm
        """
        super().__init__()
        self.layer_norm = torch.nn.GroupNorm(1, num_channels=in_channels, eps=eps)

    def forward(self, x):
        return self.layer_norm(x)


class AsyncCKCNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, positions, d_kernel=16, use_bias=True, padding_value=0):
        """
        Causal CKCNN (Romero et al., 2021) for irregular time series with asynchronous features
        :param in_channels:      Number of input channels / features
        :param hidden_channels:  Number of hidden channels / features
        :param out_channels:     Number of output channels / features
        :param positions:        Tensor of sorted positions/times at which to evaluate the convolution
        :param d_kernel:         Number of latent dimensions of the kernel network
        :param use_bias:         Whether to include an additive bias (default: True)
        :param padding_value:    Value assigned to padding (default: 0)
        """
        super(AsyncCKCNN, self).__init__()
        assert positions.size(0) % 2 != 0  # positions must be odd
        self.config = locals()

        # use AsyncCKConv to 'regularize' asynchronous, irregularly sampled input
        self._ckconv1 = AsyncCKConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            d_kernel=d_kernel,
            positions=positions,
            padding_value=padding_value,
            use_bias=use_bias
        )
        self._ckconv2 = CKConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            max_timesteps=positions.size(0),
            d_kernel=d_kernel,
            use_bias=use_bias
        )
        self._layer_norm1 = LayerNorm(hidden_channels)
        self._layer_norm2 = LayerNorm(hidden_channels)
        self._linear = torch.nn.Conv1d(hidden_channels, out_channels, kernel_size=1)
        self._relu = torch.nn.LeakyReLU()

    def kernels(self, x):
        """
        List kernels sampled according to position
        :param x:  Input signal
        :return:   Tuple of kernels from each layer
        """
        h, kernels1 = self._ckconv1(x, return_kernels=True)

        # we pass h through a layer norm in forward() thus we repeat that here
        h = self._relu(self._layer_norm1(h))
        _, kernels2 = self._ckconv2(h, return_kernels=True)
        return kernels1, kernels2

    def forward(self, x, return_last=True):
        """
        Forward pass through CKCNN
        :param x:            Input tensor of shape (batch_size, num_observations, 3),
                             where each entry is a tuple (feature, value, chart-time)
        :param return_last:  Whether to return only the output at the final timestep
        :return:             Tensor of shape (batch_size, |positions|, out_channels),
                             or, if return_last=True, (batch_size, out_channels)
        """
        h = self._relu(self._layer_norm1(self._ckconv1(x)))
        z = self._relu(self._layer_norm2(self._ckconv2(h)) + h)
        y = self._linear(z)
        return y[:, :, -1] if return_last else y


if __name__ == '__main__':
    # toy dataset of irregularly-sampled, asynchronous time series data
    x = torch.randn(size=(32, 256, 3))
    x[:, :, 0] = torch.randint(1, 46, size=(32, 256)).float()
    x[:, :, 2] = torch.cumsum(torch.rand(size=(32, 256)), dim=1)
    x[:, :25] = 0

    timesteps = torch.linspace(1, 140, 101)
    conv = AsyncCKCNN(
        in_channels=47,
        hidden_channels=56,
        out_channels=96,
        d_kernel=16,
        positions=timesteps
    )
    print('out:', conv(x).shape)

    for i, kernels in enumerate(conv.kernels(x)):
        print('Kernels layer %d:' % i, kernels.shape)