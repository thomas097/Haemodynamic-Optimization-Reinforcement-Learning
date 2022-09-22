import torch
import time


class Sine(torch.nn.Module):
    """
    Implements the Sine activation function as used by CkConv
    """
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return torch.sin(x)


class Kernel(torch.nn.Module):
    def __init__(self, hidden_units=64, use_bias=True):
        super().__init__()
        self.__kernel = torch.nn.Sequential(
            torch.nn.Linear(3, hidden_units, bias=use_bias),  # timestep + in_channel + out_channel -> hidden units
            Sine(),
            torch.nn.Linear(hidden_units, 1, bias=use_bias)
        )

    def forward(self, t):
        return self.__kernel(t)


class CkConv1D(torch.nn.Module):
    def __init__(self, in_channels, out_channels, is_causal=True, kernelnet_units=32, use_bias=True):
        super().__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._is_causal = is_causal

        self._kernel = Kernel(hidden_units=kernelnet_units,
                              use_bias=use_bias)

    def forward(self, x, t):
        # Compute Toeplitz matrix of relative positions
        dt = self._handle_relative_positions(t)

        # Sample kernel at relative positions
        kernel = self._get_kernel_as_toeplitz(dt)

        # Mask kernel if causal convolution
        if self._is_causal:
            kernel = self._causal_mask(dt) * kernel

        # Flatten input x feature-wise
        x_input = x.t().flatten()

        return torch.matmul(kernel, x_input).reshape(self._out_channels, x.shape[0]).t()

    def _causal_mask(self, dt):
        # Mask upper-triangle of kernel to ensure causal convolution
        mask = 1 - torch.triu(torch.ones(dt.shape), diagonal=1)
        return mask.repeat(self._out_channels, self._in_channels)

    def _get_kernel_as_toeplitz(self, dt):
        # Repeat Toeplitz matrix to incorporate more in_channels and out_channels
        deltas = dt.repeat(self._out_channels, self._in_channels)

        # For each repetition, create matrix with in_channel and out_channel index
        in_channels = torch.arange(self._in_channels)[None].repeat_interleave(dt.shape[0], dim=1).repeat(dt.shape[0] * self._out_channels, 1)
        out_channels = torch.arange(self._out_channels)[:, None].repeat_interleave(dt.shape[0], dim=0).repeat(1, dt.shape[0] * self._in_channels)

        # Concatenate flattened matrices to (N * N * in_channels, 2) matrix
        x_kernel = torch.concat([deltas.reshape(-1, 1),
                                 in_channels.reshape(-1, 1),
                                 out_channels.reshape(-1, 1)], dim=1)

        # Sample kernel and reshape to (|x|, |x| * in_channels)
        return self._kernel(x_kernel).reshape(*deltas.shape)

    @staticmethod
    def _handle_relative_positions(t):
        # Compute matrix of relative positions
        rep_t = torch.unsqueeze(t, 0).repeat(t.shape[0], 1)
        return rep_t - rep_t.t()


if __name__ == '__main__':
    x = torch.Tensor([[0, 1],
                      [2, 3],
                      [4, 5],
                      [6, 7],
                      [8, 9]])
    t = torch.Tensor([0, 1, 2, 3, 4])

    conv = CkConv1D(in_channels=2, out_channels=3)

    time1 = time.time_ns()
    print(conv(x, t))
    print('Time elapsed:', (time.time_ns() - time1) / 1e9)
