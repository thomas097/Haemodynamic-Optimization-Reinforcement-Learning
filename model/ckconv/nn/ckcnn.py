import time
import torch
from ckconv_layers import CKBlock


class CKCNN(torch.nn.Module):
    def __init__(self, layer_channels, d_kernel, max_timesteps, kernel_type='siren', use_residuals=True, fourier_input=True):
        """ CKConv block with layer normalization and optional residual connections (Bai et al., 2017)
        :param layer_channels:   Tuples specifying number of channels at each layer starting at the input layer,
                                 e.g. (8, 16, 4) creates a two-layer network mapping from 8 inputs to 4 outputs
                                 via a hidden layer of 16 channels
        :param d_kernel:         Dimensions of the hidden layers of the kernel network
        :param max_timesteps:    Maximum number of timesteps in input
        :param kernel_type:      Type of kernel network to use: 'siren'|'relu' (default: 'siren')
        :param use_residuals:    Whether to include a residual connection inside CKConv blocks (default: True)
        :param fourier_input:    Whether to use fourier features as input to the kernel network
        """
        super(CKCNN, self).__init__()
        self.config = locals()
        # stdout model setup
        print('Using %s kernel network' % kernel_type.upper())
        if fourier_input:
            print('Using Fourier kernel inputs')

        # CK convolution blocks
        self._blocks = []
        for i in range(len(layer_channels) - 1):
            block = CKBlock(
                in_channels=layer_channels[i],
                out_channels=layer_channels[i + 1],
                d_kernel=d_kernel,
                kernel_type=kernel_type,
                max_timesteps=max_timesteps,
                use_residuals=use_residuals,
                fourier_input=fourier_input
            )
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