import torch
import numpy as np
from input_encoding import SinCosPositionalEncoding
from transformer_layers import TransformerEncoderLayer


class CausalTransformer(torch.nn.Module):
    """ Implementation of a Transformer (Vaswani et al., 2017)
        with causal self-attention and relative positional encoding
    """
    def __init__(self, in_channels, out_channels, d_model=64, d_key=16, n_blocks=3, maxlen=256, device='cpu'):
        super(CausalTransformer, self).__init__()
        self.config = locals()
        self._maxlen = maxlen
        self._device = device

        # Input encoding
        self._input_encoding = torch.nn.Linear(in_channels, d_model)
        self._positional_encoding = torch.nn.Linear(1, d_model)

        # Transformer layers
        self._blocks = torch.nn.ModuleList()
        for _ in range(n_blocks):
            self._blocks.append(TransformerEncoderLayer(d_model=d_model, d_key=d_key))
        self._feedforward = torch.nn.Linear(d_model, out_channels)

        self._initialize()
        self.to(device)

    def _initialize(self):
        """ Initializes weights using Xavier initialization """
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def _get_positional_encoding(self, x):
        """ Compute positional encodings with a random offset (to combat position memorization) """
        assert x.size(1) <= self._maxlen

        # Compute positions [0, ..., |x| - 1] + random offset
        positions = torch.arange(x.size(1)).unsqueeze(0).unsqueeze(2).float()
        offsets = torch.randint(low=0, high=self._maxlen - x.size(1), size=(x.size(0), 1, 1))

        # Compute encoding as PositionalEncoding(Norm(positions + offset))
        off_pos = (positions + offsets) / self._maxlen
        return self._positional_encoding(off_pos).to(self._device)

    def _get_causal_mask(self, n):
        """ Upper triangular mask to mask out future positions """
        return torch.triu(torch.ones(n, n), diagonal=1).unsqueeze(0).bool().to(self._device)

    def forward(self, x, return_last=False):
        """ Forward pass through transformer """
        mask = self._get_causal_mask(x.size(1))

        y = self._input_encoding(x) + self._get_positional_encoding(x)
        for block in self._blocks:
            y = block(y, attn_mask=mask)
        return self._feedforward(y[:, -1] if return_last else y)

    def get_attention_matrices(self, x, softmax=False):
        """ Yields attention matrices of shape (batches, length, length) """
        matrices = []
        with torch.no_grad():
            mask = self._get_causal_mask(x.size(1))

            y = self._input_encoding(x) + self._get_positional_encoding(x)
            for block in self._blocks:
                # Extract attention matrix from layer
                attn = block.self_attn(y, attn_mask=mask, return_attn=True).detach()
                if softmax:
                    attn = torch.softmax(attn, dim=2)

                matrices.append(attn.numpy())
                y = block(y, attn_mask=mask)

        return np.array(matrices)
