import torch
import torch.nn.functional as F


class StateConcatenation(torch.nn.Module):
    """
    Concatenates previous K-1 states to the current state
    (Mnih et al., 2016)
    """
    def __init__(self, k=2):
        super(StateConcatenation, self).__init__()
        self.args = locals()
        self._k = k

    def forward(self, history):
        # Repeat oldest state K - |history| times if no history < K
        history_size = history.shape[1]
        if history_size < self._k:
            repeated_first_state = history[:, :1].repeat(1, self._k - history_size, 1)
            history = torch.concat([repeated_first_state, history], dim=1)

        # Drop all but last K states
        history = history[:, -self._k:]

        # Reshape histories to (batch_size, K * state_size)
        return history.reshape(-1, self._k * history.shape[2])


class CausalConv1d(torch.nn.Module):
    """ Implementation of a Causal Convolution layer
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, include_current=True):
        super(CausalConv1d, self).__init__()
        self._conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, dilation=dilation)
        self._relu = torch.nn.LeakyReLU()

        # For details see https://jmtomczak.github.io/blog/2/2_ARM.html
        self._include_current = include_current
        self._left_padding = (kernel_size - 1) * dilation + 1 * (not include_current)

    def _causal_padding(self, x):
        return F.pad(x, [self._left_padding, 0], value=0.0)

    def forward(self, x):
        padded_x = self._causal_padding(x)
        conv_x = self._conv(padded_x)
        return conv_x if self._include_current else conv_x[:, :, :-1]


class CausalCNN(torch.nn.Module):
    """
    Implementation of a Causal Convolutional Network (`CausalCNN`) based
    on dilated causal convolutions (implemented by `CausalConv1d`).
    See https://arxiv.org/pdf/1609.03499v2.pdf for details.
    """
    def __init__(self, layer_channels=(32, 32), kernel_sizes=(12,), dilations=(1,)):
        """ Constructor of the CausalConv1D """
        super(CausalCNN, self).__init__()
        self.args = locals()

        # Force kernel sizes to be odd
        self._kernel_sizes = [k + 1 if k % 2 == 0 else k for k in kernel_sizes]

        # Sanity checks
        assert len(layer_channels) - 1 == len(kernel_sizes)
        assert len(dilations) == len(kernel_sizes)

        layers = []
        for i, kernel_size in enumerate(self._kernel_sizes):
            conv = self.CausalConv1d(layer_channels[i], layer_channels[i + 1], kernel_size, dilation=dilations[i])
            layers.append(conv)
        self._model = torch.nn.Sequential(*layers)

    def forward(self, history, return_last=True):
        y = self._model(history.permute(0, 2, 1)).permute(0, 2, 1)  # Note: Conv1D expects (batch_size, in_channels, seq_length)
        return y[:, -1] if return_last else y


class LSTM(torch.nn.Module):
    """
    Implementation of a Long-Short Term Memory (LSTM) network
    (Hochreiter et al., 1997) as used by (Hausknecht et al., 2017).
    See https://arxiv.org/pdf/1507.06527.pdf for details.
    """
    def __init__(self, state_dim=46, hidden_dims=128, num_layers=1, batch_size=32):
        super(LSTM, self).__init__()
        self.args = locals()
        self._model = torch.nn.LSTM(input_size=state_dim, hidden_size=hidden_dims, num_layers=num_layers,
                                    bias=True, batch_first=True)
        self._h0 = torch.nn.Parameter(torch.randn((num_layers, 1, hidden_dims)))
        self._c0 = torch.nn.Parameter(torch.randn((num_layers, 1, hidden_dims)))

    def forward(self, history, return_last=True):
        h0 = self._h0.repeat(1, history.shape[0], 1)
        c0 = self._c0.repeat(1, history.shape[0], 1)
        h, _ = self._model(history, (h0, c0))
        return h[:, -1] if return_last else h


class GRU(torch.nn.Module):
    """
    Implementation of a Gated Recurrent Unit (GRU)
    """
    def __init__(self, state_dim=46, hidden_dims=128, num_layers=1):
        super(GRU, self).__init__()
        self.args = locals()
        self._model = torch.nn.GRU(input_size=state_dim, hidden_size=hidden_dims, num_layers=num_layers,
                                   bias=True, batch_first=True)
        self._h0 = torch.nn.Parameter(torch.randn((num_layers, 1, hidden_dims)))

    def forward(self, history, return_last=True):
        h0 = self._h0.repeat(1, history.shape[0], 1)
        h, _ = self._model(history, h0)
        return h[:, -1] if return_last else h


class EncoderDecoderLSTM(torch.nn.Module):
    """
    Feeds histories through an LSTM encoder-decoder network as
    proposed by (Peng et al., 2019).
    See https://github.com/xuefeng7/Improving-Sepsis-Treatment-Strategies for details
    """
    def __init__(self, state_dim=46, hidden_dims=128, num_layers=1):
        super(EncoderDecoderLSTM, self).__init__()
        self.args = locals()
        self._encoder = torch.nn.LSTM(input_size=state_dim, hidden_size=hidden_dims, num_layers=num_layers,
                                      bias=True, batch_first=True)
        self._decoder = torch.nn.LSTM(input_size=hidden_dims, hidden_size=state_dim, num_layers=num_layers,
                                      bias=True, batch_first=True)

    def forward(self, history, decode=False, return_last=True):
        # Feed history through encoder and optionally decoder (for pretraining)
        h, _ = self._encoder(history)
        if decode:
            h, _ = self._decoder(h)  # predict current state s_t from h0:t

        # Optionally return last encoder state
        return h[:, -1] if return_last else h


class GRUdT(torch.nn.Module):
    """
    Implementation of the GRU-∆t baseline of (Kidger et al. 2020). For details, see:
    https://proceedings.neurips.cc/paper/2020/file/4a5876b450b45371f6cfe5047ac8cd45-Paper.pdf
    """
    def __init__(self, state_dim=46, hidden_dims=128, num_layers=1):
        super(GRUdT, self).__init__()
        self.args = locals()
        self._model = torch.nn.GRU(input_size=state_dim + 1, hidden_size=hidden_dims, num_layers=num_layers,
                                   bias=True, batch_first=True)

    def forward(self, history, timesteps, return_last=True):  # -> (batch_size, num_timesteps, num_channels)
        # Compute deltas in time dimension
        deltas = torch.diff(timesteps, dim=1, prepend=timesteps[:, :1, :])  # prepend t0 again to get initial ∆t=0

        x = torch.concat([history, deltas], dim=2)
        h, _ = self._model(x)

        # Optionally return last encoder state
        return h[:, -1] if return_last else h


if __name__ == '__main__':
    BATCH_SIZE = 64
    SEQ_LENGTH = 8
    NUM_FEATURES = 12
    HIDDEN_DIMS = 128

    X = torch.randn((BATCH_SIZE, SEQ_LENGTH, NUM_FEATURES))
    print('[In]: shape =', X.shape, '\n')

    print('[Out]:')
    concat = StateConcatenation(k=10)  # not enough history!
    print('StateConcat: ', concat(X).shape)

    causal = CausalCNN(layer_channels=(12, 128), kernel_sizes=(5,), dilations=(2,))
    print('CausalCNN:   ', causal(X).shape)

    lstm = LSTM(NUM_FEATURES, HIDDEN_DIMS)
    print('LSTM:        ', lstm(X).shape)

    gru = GRU(NUM_FEATURES, HIDDEN_DIMS)
    print('GRU:         ', gru(X).shape)

    encoder_decoder = EncoderDecoderLSTM(NUM_FEATURES, HIDDEN_DIMS)
    print('Enc-Decoder: ', encoder_decoder(X).shape)

    gru_dt = GRUdT(NUM_FEATURES, HIDDEN_DIMS)
    timesteps = torch.cumsum(torch.rand(BATCH_SIZE, SEQ_LENGTH, 1), dim=1)  # (batch_size, seq_length, num_features=1)
    print('GRU-dt:      ', gru_dt(X, timesteps).shape)


