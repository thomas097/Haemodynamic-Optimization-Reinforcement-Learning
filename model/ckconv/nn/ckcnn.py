import time
import torch
from ckconv_layers import CKBlock


class CKCNN(torch.nn.Module):
    def __init__(self, layer_channels, d_kernel, max_timesteps, activation='sine', use_residual=True):
        """ CKConv block with layer normalization and optional residual connections (Bai et al., 2017)
        :param layer_channels:   Tuples specifying number of channels at each layer starting at the input layer,
                                 e.g. (8, 16, 4) creates a two-layer network mapping from 8 inputs to 4 outputs
                                 via a hidden layer of 16 channels
        :param d_kernel:         Dimensions of the hidden layers of the kernel network
        :param max_timesteps:    Maximum number of timesteps in input
        :param activation:       Type of activation to use: 'sine'|'relu' (default: 'sine')
        :param use_residual:     Whether to include a residual connection inside CKConv blocks (default: True)
        """
        super(CKCNN, self).__init__()
        self.config = locals()
        print('Using %s kernels' % activation.title())

        # CK convolution blocks
        self._blocks = []
        for i in range(len(layer_channels) - 1):
            block = CKBlock(
                in_channels=layer_channels[i],
                out_channels=layer_channels[i + 1],
                d_kernel=d_kernel,
                activation=activation,
                max_timesteps=max_timesteps,
                use_residual=use_residual)
            self._blocks.append(block)
        self._conv_layers = torch.nn.Sequential(*self._blocks)

    @property
    def kernels(self):
        return [block.ckconv.kernel for block in self._blocks]

    def forward(self, x, return_last=True):
        """ Forward pass through CKCNN
        :param x:            Input tensor of shape (batch_size, n_timesteps, in_channels)
        :param return_last:  Whether to return only the representation at the final time step
        :returns:            Output tensor of shape (batch_size, n_timesteps, out_channels)
                             if `return_last=False`, else (batch_size, out_channels)
        """
        h = x.permute(0, 2, 1)
        y = self._conv_layers(h).permute(0, 2, 1)
        return y[:, -1] if return_last else y


if __name__ == '__main__':
    X = torch.randn(64, 100, 32)

    model = CKCNN(
        layer_channels=(32, 16, 64),
        d_kernel=8,
        max_timesteps=100,
    )

    # Sanity check: time forward pass
    def timing_test(model, x, N=10):
        time1 = time.time_ns()
        for i in range(N):
            model(x)
        return (time.time_ns() - time1) / (1e9 * N)

    print('Time elapsed:', timing_test(model, X))