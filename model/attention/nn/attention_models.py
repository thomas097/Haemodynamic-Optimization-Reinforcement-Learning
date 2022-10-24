import torch
import numpy as np

from time import time_ns
from attention_layers import SelfAttentionBlock
from positional_encoding import TypeEncoding, PositionalEncoding


class CausalTransformer(torch.nn.Module):
    """
        Implementation of the Transformer (Vaswani et al., 2017).

        We treat each clinical measurement as a single 'observation' with a type (e.g. 'heart_rate'),
        a value (100) and a chart-time (in hours relative to some start-time). We encode the type by
        a learnt embedding and use sin/cos positional encoding to encode the chart-time.
        We feed the sum of embeddings (incl. the value) to a stack of encoders with a causal mask.
    """
    def __init__(self, vocab_size, d_model, nhead=1, batch_size=32, max_len=1200, mask_layer=True):
        super().__init__()
        # Positional- and ItemEncoding layers
        self._pos_encoding = PositionalEncoding(d_model=d_model, batch_size=batch_size)
        self._type_encoding = TypeEncoding(vocab_size=vocab_size + 1, embedding_dim=d_model)  # +1 for padding (0)

        # TransformerEncoder layer
        self._encoder = SelfAttentionBlock(d_model=d_model, nhead=nhead, max_len=max_len)

        self._max_len = max_len  # Protects against overflow
        self._mask_layer = mask_layer

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, X, return_last=True):
        # Unpack input as sequence of (type, value, timestep) tuples of at most max_len
        x_type = X[:, -self._max_len:, 0]
        x_value = X[:, -self._max_len:, 1]
        x_tstep = X[:, -self._max_len:, 2]

        # Construct input Tensor
        type_embedding = self._type_encoding(x_type.long())
        pos_embedding = self._pos_encoding(x_tstep)

        inputs = type_embedding + pos_embedding
        inputs[:, :, -1] = x_value  # Note: last channel is least corrupted by PE

        # Mask out padding
        padding_mask = (x_type > 0).bool().unsqueeze(1).repeat(1, x_type.shape[1], 1)  # TODO: mask out inter-type attentions

        out = self._encoder(inputs, mask=padding_mask)
        return out[:, -1] if return_last else out
