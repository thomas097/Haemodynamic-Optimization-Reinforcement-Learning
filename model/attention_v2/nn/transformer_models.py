import torch
import numpy as np
from time import time_ns

from input_encoding import CategoricalEncoding, LinearPositionalEncoding, SinCosPositionalEncoding
from transformer_layers import SelfAttentionHead, TransformerEncoderLayer


class CausalTransformer(torch.nn.Module):
    """ Implementation of a Causal Transformer (Vaswani et al., 2017) with
        a learnable Self-Attention head.
    """
    def __init__(self, vocab, out_channels, pos_dims=32, type_dims=32, d_key=32, n_queries=8,
                 truncate=256, padding_token=0, causal=True, attention_head=False):
        super(CausalTransformer, self).__init__()
        self.config = locals()
        self._attention_head = attention_head
        self._padding_token = padding_token
        self._maxlen = truncate
        self._causal = causal

        # Input encoding
        self._pos_encoder = SinCosPositionalEncoding(pos_dims)
        self._type_encoder = CategoricalEncoding(type_dims, vocab_size=len(vocab))

        # Transformer layers
        d_model = pos_dims + type_dims + 1
        self._encoder = TransformerEncoderLayer(d_model, d_key=d_key)
        if attention_head:
            self._head = SelfAttentionHead(d_model, out_channels, n_queries=n_queries, d_key=d_key)
        else:
            self._head = torch.nn.Linear(d_model, out_channels)

    @staticmethod
    def _causal_attn_mask(t):
        return (t.unsqueeze(2) - t.unsqueeze(1)) > 0  # -> (batch_size, n_timesteps, n_timesteps)

    def _padding_key_mask(self, x_type):
        return (x_type == self._padding_token).unsqueeze(2)  # -> (batch_size, n_timesteps, dk=1)

    def forward(self, x):
        # Truncate input sequence x to at most self._maxlen
        x_type = x[:, -self._maxlen:, 0].long()
        x_val = x[:, -self._maxlen:, 1]
        x_pos = x[:, -self._maxlen:, 2]

        # Construct input Tensor encoding type and value
        encoded_type = self._type_encoder(x_type)
        encoded_pos = self._pos_encoder(x_pos)
        inputs = torch.concat([encoded_type, encoded_pos, x_val.unsqueeze(2)], dim=2)

        # Mask zero-padding future positions using Boolean tensors
        key_padding_mask = self._padding_key_mask(x_type)
        src_mask = self._causal_attn_mask(x_pos) if self._causal else None

        h = torch.relu(self._encoder(inputs, src_mask, key_padding_mask))
        if self._attention_head:
            return self._head(h, key_padding_mask)
        else:
            return self._head(h[:, -1])


if __name__ == '__main__':
    # Create toy dataset
    X = torch.rand(64, 512, 3)                             # 1 = value
    X[:, :, 0] = torch.randint(0, 47, (64, 512)).float()   # 0 = type
    X[:, :, 2] = torch.cumsum(torch.rand(64, 512), dim=1)  # 2 = timestep
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
