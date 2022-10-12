import torch
from time import time_ns


INF = 1e32


class SelfAttention(torch.nn.Module):
    """ Implementation of a Self-Attention layer (Vaswani et al., 2017)
    """
    def __init__(self, in_channels, out_channels, dk=16, is_causal=True):
        super().__init__()
        self._Q = torch.nn.Linear(in_channels, dk)
        self._K = torch.nn.Linear(in_channels, dk)
        self._V = torch.nn.Linear(in_channels, out_channels)
        self._dk = torch.tensor(dk)
        self._is_causal = is_causal

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def _causal_mask(self, attn_matrix):
        # Replace upper triangle of attention matrix by -inf
        mask = torch.tril(torch.ones((attn_matrix.shape[1], attn_matrix.shape[1]))).to(self._device)
        return mask * attn_matrix + (1 - mask) * -INF

    def forward(self, x, mask=None):       
        Q = self._Q(x)
        K = self._K(x)
        V = self._V(x)

        attn_matrix = torch.bmm(Q, torch.transpose(K, 1, 2)) / torch.sqrt(self._dk)

        # Mask future positions if `is_causal=True`
        if self._is_causal:
            attn_matrix = self._causal_mask(attn_matrix)

        # When Boolean `mask` is given, drop disallowed attentions (False -> -Inf)
        if mask is not None:
            mask.to(self._device)
            attn_matrix = mask * attn_matrix + ~mask * -INF

        return torch.bmm(torch.softmax(attn_matrix, dim=2), V)


class MultiHeadSelfAttention(torch.nn.Module):
    """ Implementation of Multi-Head Self-Attention (Vaswani et al., 2017)
    """
    def __init__(self, in_channels, out_channels, k=4):
        super().__init__()
        # List of K self-attention heads and reduction layer
        self._heads = torch.nn.ModuleList([SelfAttention(in_channels, out_channels) for _ in range(k)])
        self._linear = torch.nn.Linear(k * out_channels, out_channels)

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, x, mask=None):
        hidden = torch.concat([head(x, mask=mask) for head in self._heads], dim=2)
        # Return output straight away if K=1
        if len(self._heads) == 1:
            return hidden
        return self._linear(hidden)


class SelfAttentionBlock(torch.nn.Module):
    def __init__(self, d_model, k=4):
        super().__init__()
        self._attn = MultiHeadSelfAttention(d_model, d_model, k=k)
        self._layer_norm = torch.nn.LayerNorm(d_model)

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, x, mask=None):
        # Self-Attention with residual connections and LayerNorm
        return self._layer_norm(x + self._attn(x, mask=mask))


if __name__ == '__main__':
    X = torch.randn((1, 1700, 32))  # -> (batch_size, seq_length, num_features)
    print('In:', X.shape)

    attn = SelfAttentionBlock(d_model=32, k=2)

    # Timing test
    time1 = time_ns()
    out = attn(X)
    time2 = time_ns()
    print('Out:', out)

    print('Time elapsed:', (time2 - time1) / 1e9)
