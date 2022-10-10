import torch


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
        self._inf = 1e32

    def _causal_mask(self, attn):
        # Replace upper triangle of attention matrix by -inf
        mask = torch.tril(torch.ones((attn.shape[1], attn.shape[1])))
        return mask * attn + (1 - mask) * -self._inf

    def forward(self, x, mask=None):       
        Q = self._Q(x)
        K = self._K(x)
        V = self._V(x)

        attn = torch.bmm(Q, torch.transpose(K, 1, 2)) / torch.sqrt(self._dk)

        # Mask future positions if `is_causal=True`
        if self._is_causal:
            attn = self._causal_mask(attn)

        # When Boolean `mask` is given, drop disallowed attentions (False -> -Inf)
        if mask is not None:
            attn = mask * attn + ~mask * -self._inf

        return torch.bmm(torch.softmax(attn, dim=2), V)


class MultiHeadAttention(torch.nn.Module):
    """ Implementation of Multi-Head Self-Attention (Vaswani et al., 2017)
    """
    def __init__(self, in_channels, out_channels, K=4):
        super().__init__()
        # List of K self-attention heads and reduction layer
        self._heads = torch.nn.ModuleList([SelfAttention(in_channels, out_channels) for _ in range(K)])
        self._linear = torch.nn.Linear(K * out_channels, out_channels)

    def forward(self, x, mask=None):
        hidden = torch.concat([head(x, mask=mask) for head in self._heads], dim=2)
        # Return output straight away if K=1
        if len(self._heads) == 1:
            return hidden
        return self._linear(hidden)


if __name__ == '__main__':
    X = torch.randn((1, 1000, 32))  # -> (batch_size, seq_length, num_features)
    print('In:', X.shape)

    attn = MultiHeadAttention(32, 64, K=1)
    print('Out:', attn(X).shape)
