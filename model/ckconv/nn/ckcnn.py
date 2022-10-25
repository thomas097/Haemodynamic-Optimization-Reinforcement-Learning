import time
import torch
from ckconv_layers import CKBlock


class CKCNN(torch.nn.Module):
    def __init__(self, layer_channels=(64, 128), kernel_dims=32, max_timesteps=100):
        super().__init__()
        self._blocks = []
        for i in range(len(layer_channels) - 1):
            self._blocks.append(CKBlock(layer_channels[i], layer_channels[i + 1], kernel_dims, max_timesteps))
        self._model = torch.nn.Sequential(*self._blocks)

        # Use GPU when available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    @property
    def kernels(self):
        return [block.ckconv.kernel for block in self._blocks]

    def forward(self, x):  # <- (batch_size, seq_length, in_channels)
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