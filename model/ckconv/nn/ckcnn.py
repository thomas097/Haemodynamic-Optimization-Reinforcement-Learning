import time
import torch
from ckconv_layers import CKBlock


class CKCNN(torch.nn.Module):
    def __init__(self, layer_channels=(64, 128), kernel_dims=32, max_timesteps=100):
        super(CKCNN, self).__init__()
        self.config = locals()

        self._blocks = []
        for i in range(len(layer_channels) - 1):
            self._blocks.append(CKBlock(layer_channels[i], layer_channels[i + 1], kernel_dims, max_timesteps))
        self._model = torch.nn.Sequential(*self._blocks)

        # Use GPU when available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

        # Learnt padding vector to change zero-padding in input
        self._padding = torch.nn.Parameter(torch.randn(layer_channels[0]).unsqueeze(0).unsqueeze(0)).to(self._device)

    @property
    def kernels(self):
        return [block.ckconv.kernel for block in self._blocks]

    def _proper_padding(self, x):
        mask = torch.all(x == 0, dim=2, keepdim=True)
        padding = self._padding.repeat(x.shape[0], x.shape[1], 1)
        return mask * padding + (~mask) * x

    def forward(self, x):  # <- (batch_size, seq_length, in_channels)
        x = self._proper_padding(x)
        y = self._model(x.permute(0, 2, 1))
        return y[:, :, -1]


if __name__ == '__main__':
    X = torch.randn(64, 10, 32)

    model = CKCNN(layer_channels=(32, 64))

    # Sanity check: time forward pass
    def timing_test(model, x, N=10):
        time1 = time.time_ns()
        for i in range(N):
            model(x)
        return (time.time_ns() - time1) / (1e9 * N)

    print('Time elapsed:', timing_test(model, X))