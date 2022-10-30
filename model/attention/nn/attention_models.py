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
    def __init__(self, vocab_size, d_model, out_channels, nheads=1, dk=16, action_id=45, truncate=256):
        super(CausalTransformer, self).__init__()
        self.config = locals()
        self._truncate = truncate
        self._action_id = action_id

        # Positional- and TypeEncoding layers + Transformer Encoder
        self._pos_encoding = PositionalEncoding(d_model)
        self._type_encoding = TypeEncoding(d_model, vocab_size=vocab_size + 1)  # vocab + 1 for padding
        self._value_encoding = torch.nn.Linear(1, d_model)

        self._encoder1 = SelfAttentionBlock(d_model, d_model, nheads=nheads, dk=dk, maxlen=truncate)
        self._encoder2 = SelfAttentionBlock(d_model, out_channels, nheads=nheads, dk=dk, maxlen=truncate)

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, x, return_last=True):
        # Unpack input as sequence of (type, value, timestep) tuples of at most max_len
        x_type = x[:, -self._truncate:, 0]
        x_value = x[:, -self._truncate:, [1]]
        x_pos = x[:, -self._truncate:, 2]

        # Construct input Tensor x encoding type, position and value
        te = self._type_encoding(x_type.long())
        pe = self._pos_encoding(x_pos)
        ve = self._value_encoding(x_value)
        inputs = te + pe + ve

        # Mask zero-padding/old 'action' tokens using Boolean mask of shape (batch_size, seq_len, dk=1) with True -> padding
        key_mask = torch.logical_or(x_type == 0, x_type == self._action_id).unsqueeze(2).to(self._device)
        key_mask[:, -1, :] = False  # Don't mask out final 'action' state

        # Forward pass through transformer
        hidden = self._encoder1(inputs, key_mask=key_mask)
        return self._encoder2(hidden, key_mask=key_mask, return_last=return_last)