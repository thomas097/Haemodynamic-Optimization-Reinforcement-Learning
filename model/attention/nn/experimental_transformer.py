import torch
import numpy as np
from time import time_ns

from input_encoding import CategoricalEncoding, LinearPositionalEncoding, SinCosPositionalEncoding
from transformer_layers import TransformerEncoderLayer


class CausalTransformer(torch.nn.Module):
    """ Implementation of a Transformer (Vaswani et al., 2017)
        with causal self-attention
    """
    def __init__(self, vocab, out_channels, d_model=64, d_key=16, n_blocks=3, truncate=0, dropout=0.1, padding_value=0):
        super(CausalTransformer, self).__init__()
        self.config = locals()
        self._padding_value = padding_value
        self._maxlen = truncate

        # Input encoding
        self._type_encoder = CategoricalEncoding(d_model, vocab_size=len(vocab))
        self._value_encoder = torch.nn.Linear(1, d_model)
        self._pos_encoder = SinCosPositionalEncoding(d_model)

        # Transformer layers
        self._blocks = torch.nn.ModuleList()
        for _ in range(n_blocks):
            self._blocks.append(TransformerEncoderLayer(d_model=d_model, d_key=d_key, dropout=dropout))
        self._feedforward = torch.nn.Linear(d_model, out_channels)

    @staticmethod
    def _create_causal_mask(t):
        return (t - t.permute(0, 2, 1)) < 0

    def _create_padding_mask(self, x_type):
        return (x_type == self._padding_value).unsqueeze(1)

    def forward(self, x, return_last=False):
        # Truncate input sequence x to at most self._maxlen
        x_type = x[:, -self._maxlen:, 0].long()
        x_val = x[:, -self._maxlen:, 1].unsqueeze(2).float()
        x_pos = x[:, -self._maxlen:, 2].unsqueeze(2).float()

        # Construct input Tensor encoding type and value
        e_pos = self._pos_encoder(x_pos)
        e_val = self._value_encoder(x_val)
        e_type = self._type_encoder(x_type)
        x = e_pos + e_val + e_type

        # Mask zero-padding and future positions using Boolean masks
        key_padding_mask = self._create_padding_mask(x_type)
        causal_attn_mask = self._create_causal_mask(x_pos)
        attn_mask = torch.logical_or(key_padding_mask, causal_attn_mask)

        for block in self._blocks:
            x = block(x, attn_mask=attn_mask)
        return self._feedforward(x[:, -1] if return_last else x)


if __name__ == '__main__':
    # Create toy dataset
    X = torch.rand(64, 512, 3)                             # 1 = value
    X[:, :, 0] = torch.randint(1, 47, (64, 512)).float()   # 0 = type
    X[:, :, 2] = torch.cumsum((torch.rand(64, 512) * 3 < 2).float(), dim=1)  # 2 = timestep
    X[:, :40] = 0  # Mask first 40 as 'padding'

    t = CausalTransformer(vocab=np.arange(48), out_channels=64, d_key=16, truncate=512)

    # Timing test
    y = None
    N = 20
    time1 = time_ns()
    for i in range(N):
        y = t(X)
    time2 = time_ns()
    print('time elapsed:', (time2 - time1) / (N * 1e9))

    print('Out:', y.shape)
