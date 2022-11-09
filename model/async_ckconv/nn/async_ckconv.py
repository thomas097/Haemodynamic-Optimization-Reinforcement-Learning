import torch
import torch.nn.functional as F
from torch.distributions.normal import Normal
import numpy as np
import matplotlib.pyplot as plt


class Sine(torch.nn.Module):
    def __init__(self, in_channels, out_channels, omega_0=1.0, use_bias=True):
        """
        Implements a Sine layer (Romero et al., 2021) with tunable frequency omega_0
        For details: https://arxiv.org/pdf/2102.02611.pdf
        :param in_channels:   Number of input channels / features
        :param out_channels:  Number of output channels / features
        :param omega_0:       Initial guess of frequency parameter
        :param use_bias:      Whether to include an additive bias
        """
        super(Sine, self).__init__()
        self._linear = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1), bias=use_bias)
        self._omega_0 = torch.nn.Parameter(torch.tensor(omega_0))

    def forward(self, x):
        """
        Forward pass through Sine layer
        :param x:  Feature map of shape (batch_size, in_channels, height, width)
        :returns:  Feature map of shape (batch_size, out_channels, height, width)
        """
        return torch.sin(self._omega_0 * self._linear(x))


class KernelNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, omega_0=1.0, use_bias=True):
        """
        SIREN-based kernel network
        :param in_channels:   Number of input channels / features
        :param out_channels:  Number of hidden channels / features
        :param out_channels:  Number of output channels / features
        :param omega_0:       Initial guess of the frequency parameter
        :param use_bias:      Whether to include an additive bias
        """
        super(KernelNet, self).__init__()
        self._ff = torch.nn.Sequential(
            Sine(in_channels, hidden_channels, omega_0=omega_0, use_bias=use_bias),
            Sine(hidden_channels, hidden_channels, omega_0=omega_0, use_bias=use_bias),
            torch.nn.Conv2d(hidden_channels, out_channels, kernel_size=(1, 1), bias=use_bias)
        )
        self._initialize(omega_0)

    def _initialize(self, omega_0):
        """
        Initializes the weights of the SIREN
        :param omega_0: Initial guess of frequency parameter
        """
        first_layer = True
        for (i, layer) in enumerate(self.modules()):
            if isinstance(layer, torch.nn.Conv2d):  # Note: Sine's are 2D here as they work on |t|x|t| distance matrices!
                if first_layer:
                    layer.weight.data.uniform_(-1, 1)
                    first_layer = False
                else:
                    val = np.sqrt(6.0 / layer.weight.shape[1]) / omega_0
                    layer.weight.data.uniform_(-val, val)

    def forward(self, x):
        """
        Forward pass through Sine layer
        :param x:  Feature map of shape (batch_size, in_channels, height, width)
        :returns:  Feature map of shape (batch_size, out_channels, height, width)
        """
        return self._ff(x)


class AsyncCKConv(torch.nn.Module):
    def __init__(self, in_channels, out_channels, positions, d_kernel, use_bias=True, padding_value=0):
        """
        Causal CK convolutional layer for irregular time series with asynchronous features
        :param in_channels:      Number of input features
        :param out_channels:     Number of output features
        :param positions:        Tensor of sorted positions/times at which to evaluate the convolution
        :param d_kernel:         Number of latent dimensions of the kernel network
        :param use_bias:         Whether to include an additive bias (default: True)
        :param padding_value:    Value assigned to padding (default: 0)
        """
        super(AsyncCKConv, self).__init__()
        # we sample convolution weights (equal to |out_channels|) from a kernel net
        # with a tuple of the form (in_channel: [int], rel_pos: [float])
        self._net = KernelNet(in_channels=in_channels + 1,
                              hidden_channels=d_kernel,
                              out_channels=out_channels)
        # learn bias directly
        if use_bias:
            self._bias = torch.nn.Parameter(torch.randn(1, out_channels, 1))
        else:
            self.register_buffer('_bias', torch.zeros(1, out_channels, 1), persistent=True)

        self._in_channels = in_channels
        self._positions = positions.unsqueeze(0)  # add pseudo-batch
        self._padding_value = padding_value

    def _handle_kernel_input(self, x_feat, x_time):
        """
        Prepares (x_feat, x_time) pairs for input to the kernel network
        :param x_feat:  LongTensor of feature indices of shape (batch_size, num_observations)
        :param x_time:  FloatTensor of time points of shape (batch_size, num_observations)
        :returns:       Kernel input tensor of shape (batch_size, num_features + 1, |timesteps|, num_observations)
        """
        # compute relative distances -1 < d < 1 between elements of tensors self._positions and x_time
        time_deltas = (x_time.unsqueeze(1) - self._positions.unsqueeze(2)).unsqueeze(3)
        time_deltas = time_deltas / torch.max(self._positions)

        # computes a one-hot encoding of the in-channels
        one_hot_channels = F.one_hot(input=x_feat, num_classes=self._in_channels)
        one_hot_channels = one_hot_channels.unsqueeze(1).repeat(1, self._positions.size(1), 1, 1)

        return torch.concat([one_hot_channels, time_deltas], dim=3).permute(0, 3, 1, 2)

    def _causal_mask(self, x_time):
        """
        Computes a causal mask for kernel weights
        :param x_time:  FloatTensor of time points of shape (batch_size, num_observations)
        :returns:       Boolean mask of shape (batch_size, 1, |timesteps|, num_observations)
                        where elements assigned True are masked out
        """
        time_deltas = (x_time.unsqueeze(1) - self._positions.unsqueeze(2)).unsqueeze(1)
        return time_deltas > 0

    def _padding_mask(self, x):
        """
        Computes a padding mask for kernel weights
        :param x:  Tensor of feature indices of shape (batch_size, num_observations, 3)
        :returns:  Boolean mask of shape (batch_size, 1, 1, num_observations) where
                   entries assigned True are masked out
        """
        # if feature, value and timestep are all zero => assume padding
        return (torch.all(x == self._padding_value, dim=2)).unsqueeze(1).unsqueeze(2)

    @staticmethod
    def _estimate_inv_density(x_feat, x_time, h=1):
        """
        Estimates for each observation x_i in x, a kernel-density estimate of the local density
        around x_i to discount clusters of close points.
        For details, see wikipedia: https://en.wikipedia.org/wiki/Kernel_density_estimation
        :param x_feat:  LongTensor of feature indices of shape (batch_size, num_observations)
        :param x_time:  FloatTensor of time points of shape (batch_size, num_observations)
        :param h:       Scaled kernel parameter (higher is smoother)
        :returns:       Density estimates of shape (batch_size, 1, num_observations, 1)
        """
        # compute scaled distance between all points of all features
        scaled_dists = (x_time.unsqueeze(1) - x_time.unsqueeze(2)) / h

        # compute value of density K for distance (assuming normal kernel)
        normal = Normal(torch.tensor([0.0]), torch.tensor([1.0]))
        k = torch.exp(normal.log_prob(scaled_dists))

        # mask out cross-feature / channel distances
        within_feat = (x_feat.unsqueeze(1) - x_feat.unsqueeze(2)) == 0
        k[~within_feat] = 0

        # estimate within-feature inverse density
        N = torch.sum(within_feat, dim=1, keepdim=True)
        density = torch.sum(k, dim=1, keepdim=True) / (N * h)
        return 1 / density.unsqueeze(3)

    def forward(self, x, return_kernels=False):
        """
        Forward pass through AsyncCKConv
        :param x:              Input tensor of shape (batch_size, num_observations, 3), where each
                               entry represents an observation of the form (feature, value, time)
        :param return_kernels: Whether to include the sampeled kernels in the output (default: False)
        :returns:              Tensor of shape (batch_size, |timesteps|, out_channels)
        """
        # unpack (feature, value, time) tuples
        x_feat = x[:, :, 0].long()
        x_vals = x[:, :, 1].unsqueeze(1).unsqueeze(3).float()
        x_time = x[:, :, 2].float()

        # sample weights from kernel net
        kernel = self._net(self._handle_kernel_input(x_feat, x_time))

        # mask out future positions and padding positions
        mask = torch.logical_or(self._causal_mask(x_time),
                                self._padding_mask(x))
        kernel = kernel.masked_fill(mask=mask, value=0.0)

        # scale values by density around observation in same channel
        # to de-bias the convolution operation, see: https://arxiv.org/abs/2102.02611
        x_vals *= self._estimate_inv_density(x_feat, x_time)

        # convolution as matrix-product with sampled kernel
        out = torch.matmul(kernel, x_vals)[:, :, :, 0] + self._bias

        return (out, kernel) if return_kernels else out


if __name__ == '__main__':
    # toy dataset of irregularly-sampled, asynchronous time series data
    x = torch.randn(size=(32, 256, 3))
    x[:, :, 0] = torch.randint(1, 46, size=(32, 256)).float()
    x[:, :, 2] = torch.cumsum(torch.rand(size=(32, 256)), dim=1)
    x[:, :25] = 0

    timesteps = torch.linspace(1, torch.max(x[0, :, 2]), 100)
    conv = AsyncCKConv(in_channels=47,
                       out_channels=96,
                       d_kernel=16,
                       positions=timesteps)

    print('out:', conv(x).shape)

    # what are we seeing?
    import matplotlib.pyplot as plt

    y = conv(x)[0].detach().numpy()
    plt.subplot(2, 1, 1)
    plt.imshow(y, aspect='auto')
    plt.subplot(2, 1, 2)
    plt.plot(np.mean(y, axis=0))
    plt.show()