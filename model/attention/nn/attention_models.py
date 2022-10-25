import torch
import numpy as np

from time import time_ns
from attention_layers import SelfAttentionBlock
from positional_encoding import TypeEncoding, PositionalEncoding


class CausalTransformer(torch.nn.Module):
    """
        Implementation of the Transformer (Vaswani et al., 2017) with Causal Attention.

        We treat each clinical measurement as a single 'observation' with a type (e.g. 'heart_rate'),
        a value (e.g. 100) and a chart time (in hours relative to a start-time). We encode types
        by learnt embeddings and use sin/cos positional encoding to encode the chart-time.
        We feed the sum of embeddings (incl. the value) to a stack of causal self-attention encoders.
    """
    def __init__(self, vocab_size, d_model, nheads=1, dk=16, truncate=128, mask_layer=True):
        super().__init__()
        # Positional- and TypeEncoding layers + Transformer Encoder
        self._pos_encoding = PositionalEncoding(embedding_dim=d_model)
        self._type_encoding = TypeEncoding(embedding_dim=d_model, vocab_size=vocab_size + 1)  # vocab + 1 for padding
        self._encoder = SelfAttentionBlock(d_model=d_model, nheads=nheads, dk=dk, maxlen=truncate)

        self._truncate = truncate
        self._mask_layer = mask_layer

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, X, return_last=True):
        # Unpack input as sequence of (type, value, timestep) tuples of at most max_len
        x_type = X[:, -self._truncate:, 0]
        x_value = X[:, -self._truncate:, 1]
        x_pos = X[:, -self._truncate:, 2]

        # Construct input Tensor x encoding type, position and value
        te = self._type_encoding(x_type.long())
        pe = self._pos_encoding(x_pos)
        x = te + pe
        x[:, :, -1] = x_value  # Note: last channel is least corrupted by PE

        # Mask out zero-padding using Boolean mask of shape (batch_size, seq_len, dk=1) with True -> padding
        padding_mask = (x_type == 0).bool().unsqueeze(2).to(self._device)

        out = self._encoder(x, padding_mask=padding_mask)
        return out[:, -1] if return_last else out
