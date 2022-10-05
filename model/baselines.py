import torch
from torch.autograd import Variable


class StateConcatenation(torch.nn.Module):
    """
    Concatenates previous K-1 states to the current state
    """
    def __init__(self, K=2):
        super().__init__()
        self._K = K

    def forward(self, histories):
        # Drop all but last K states
        short_histories = histories[:, -self._K:]

        # Reshape histories to (batch_size, K * state_size)
        return short_histories.reshape(histories.shape[0], -1)


class LSTMAutoEncoder(torch.nn.Module):
    """ Feeds histories through an LSTM encoder-decoder network as first
        proposed in (Peng et al., 2019) (note that the code of Peng et al.
        simply mimics a copying problem as `hidden_dim > state_dim`).
        For details: https://github.com/xuefeng7/Improving-Sepsis-Treatment-Strategies
    """
    def __init__(self, state_dim=46, hidden_dims=128):
        super().__init__()
        self._encoder = torch.nn.LSTM(input_size=state_dim, hidden_size=hidden_dims, num_layers=1,
                                      bias=True, batch_first=True)
        self._decoder = torch.nn.LSTM(input_size=hidden_dims, hidden_size=state_dim, num_layers=1,
                                      bias=True, batch_first=True)

    def forward(self, histories, decode=False):  # -> (batch_size, num_timesteps, num_channels)
        encoder_states, _ = self._encoder(histories)
        if decode:
            return self._decoder(encoder_states)[0]  # "predict current state s_t given embeddings up until time t"
        else:
            return encoder_states[:, -1]


if __name__ == '__main__':
    BATCH_SIZE = 64
    SEQ_LENGTH = 18
    NUM_FEATURES = 56
    HIDDEN_DIMS = 128

    X = torch.randn((BATCH_SIZE, SEQ_LENGTH, NUM_FEATURES))
    print('Input shape:', X.shape, '\n')

    # Baselines
    concat = StateConcatenation()
    print('StateConcat output:', concat(X).shape)

    autoencoder = LSTMAutoEncoder(NUM_FEATURES, HIDDEN_DIMS)
    print('AutoEncoder output:', autoencoder(X).shape)


