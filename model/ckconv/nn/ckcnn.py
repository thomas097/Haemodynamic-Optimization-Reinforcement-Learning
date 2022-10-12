import time
import torch
from ckconv_layers import CKBlock


class CKCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_blocks=1):
        super().__init__()
        self._linear = torch.nn.Linear(in_channels, out_channels)

        blocks = [CKBlock(out_channels, out_channels) for _ in range(num_blocks)]
        self._stack = torch.nn.Sequential(*blocks)

        # Use GPU when available
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

    def forward(self, x, return_last=True):
        self._linear(x)
        y = self._stack(self._linear(x))
        return y[:, -1] if return_last else y


if __name__ == '__main__':
    X = torch.randn(64, 10, 32)

    model = CKCNN(32, 64, num_blocks=1)

    # Sanity check: time forward pass
    def timing_test(model, x, N=10):
        time1 = time.time_ns()
        for i in range(N):
            model(x)
        return (time.time_ns() - time1) / (1e9 * N)

    print('Time elapsed:', timing_test(model, X))