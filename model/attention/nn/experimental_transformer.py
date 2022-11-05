import torch
import numpy as np
import torch.nn.functional as F
from transformer_layers import TransformerEncoderLayer


class CausalTransformer(torch.nn.Module):
    """ Implementation of a Transformer (Vaswani et al., 2017)
        with causal self-attention
    """
    def __init__(self, vocab_size, out_channels, d_model=64, d_key=16, n_blocks=3, truncate=0, padding_value=0):
        super(CausalTransformer, self).__init__()
        self.config = locals()

        self._input_encoding = torch.nn.Sequential(
            torch.nn.Linear(vocab_size + 1, d_model),  # measurement type + value
            torch.nn.LeakyReLU(),
            torch.nn.Linear(d_model, d_model)
        )

        self._blocks = torch.nn.ModuleList()
        for _ in range(n_blocks):
            self._blocks.append(TransformerEncoderLayer(d_model=d_model, d_key=d_key))
        self._feedforward = torch.nn.Linear(d_model, out_channels)

        self._padding_value = padding_value
        self._maxlen = truncate

    @staticmethod
    def _causal_mask(x):
        """ Computes an upper triangular mask to mask out future positions """
        t = x[:, :, 2].unsqueeze(2)
        return (t - t.permute(0, 2, 1)) < 0

    def _padding_mask(self, x):
        """ Computes block mask to mask out padding tokens """
        t = x[:, :, 0].long().unsqueeze(2)
        return (t == self._padding_value).unsqueeze(1)

    def _encode_inputs(self, x):
        """ Encodes a (type, value) pair using a 2-layer neural network """
        x_val = x[:, :, 1].unsqueeze(2)
        x_type = F.one_hot(x[:, :, 0].long(), num_classes=self._vocab_size)
        return torch.concat([x_val, x_type], dim=2)

    def forward(self, x, return_last=False):
        """ Forward pass through network """
        # Truncate long sequences
        x = x[:, -self._maxlen:].float()

        # Mask zero-padding and future positions using Boolean masks
        attn_mask = torch.logical_or(self._padding_mask(x),
                                     self._causal_mask(x))

        import matplotlib.pyplot as plt
        plt.imshow(attn_mask[0].detach().numpy())
        plt.show()

        y = self._encode_inputs(x)
        for block in self._blocks:
            y = block(y, attn_mask=attn_mask)
        return self._feedforward(y[:, -1] if return_last else y)
