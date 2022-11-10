import torch
import torch.nn.functional as F
from async_embedding import AsyncFeatureExtraction
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
    def __init__(self, in_channels, hidden_channels, out_channels, positions, d_kernel=16, d_embedding=8, use_bias=True, padding_value=0):
        """
        Causal CKCNN (Romero et al., 2021) for irregular time series with asynchronous features
        :param in_channels:      Number of input channels / features
        :param hidden_channels:  Number of hidden channels / features
        :param out_channels:     Number of output channels / features
        :param positions:        Tensor of sorted positions/times at which to evaluate the convolution
        :param d_kernel:         Number of latent dimensions of the kernel network
        :param d_kernel:         Number of latent dimensions of the feature / channel embeddings (default: 8)
        :param use_bias:         Whether to include an additive bias (default: True)
        :param padding_value:    Value assigned to padding (default: 0)
        """
        super(AsyncCKCNN, self).__init__()
        assert positions.size(0) % 2 != 0  # positions must be odd
        self.config = locals()

        # use AsyncCKConv to 'regularize' asynchronous, irregularly sampled input
        self._feature_extr = AsyncFeatureExtraction(
            in_channels=in_channels,
            out_channels=hidden_channels,
            out_positions=positions,
            d_embedding=d_embedding,
        )
        self._ckconv = CKConv(
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
        """ Returns weights of CKConv kernel
        :param x:  Input tensor of shape (batch_size, num_observations, 3),
                   where each entry is a tuple (feature, value, chart-time)
        :returns:  Weight matrix of size (out_channels, in_channels, 1)
        """
        h = self._relu(self._layer_norm1(self._feature_extr(x)))
        _, kernel = self._ckconv(h, return_kernels=True)
        return kernel

    def forward(self, x, return_last=True):
        """
        Forward pass through CKCNN
        :param x:            Input tensor of shape (batch_size, num_observations, 3),
                             where each entry is a tuple (feature, value, chart-time)
        :param return_last:  Whether to return only the output at the final timestep
        :return:             Tensor of shape (batch_size, |positions|, out_channels),
                             or, if return_last=True, (batch_size, out_channels)
        """
        h = self._relu(self._layer_norm1(self._feature_extr(x)))
        z = self._relu(self._layer_norm2(self._ckconv(h)) + h)
        y = self._linear(z)
        return y[:, :, -1] if return_last else y


if __name__ == '__main__':
    # toy dataset of irregularly-sampled, asynchronous time series data
    x = torch.randn(size=(32, 256, 3))
    x[:, :, 0] = torch.randint(1, 46, size=(32, 256)).float()
    x[:, :, 2] = torch.cumsum(torch.rand(size=(32, 256)), dim=1)
    x[:, :25] = 0

    conv = AsyncCKCNN(
        in_channels=47,
        hidden_channels=56,
        out_channels=96,
        d_kernel=16,
        positions=torch.arange(1, 72, 72).float()
    )

    print('go!')
    print('out:', conv(x).shape)
    print('kernel:', conv.kernels(x).shape)