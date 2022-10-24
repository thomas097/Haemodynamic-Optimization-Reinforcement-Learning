import torch
from time import time_ns


INF = 1e32


class SelfAttention(torch.nn.Module):
    """ 
        Implementation of a Self-Attention layer (Vaswani et al., 2017)
        with causal attention masking
    """
    def __init__(self, in_channels, out_channels, max_len, dk=16, is_causal=True):
        super().__init__()
        self._Q = torch.nn.Linear(in_channels, dk)
        self._K = torch.nn.Linear(in_channels, dk)
        self._V = torch.nn.Linear(in_channels, out_channels)
        self._dk = torch.tensor(dk)

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

        # Precompute causal mask of (batch_size, max_len, max_len)
        if is_causal:
            self._causal_mask = torch.tril(torch.ones(max_len, max_len)).bool().unsqueeze(0).to(self._device)
        else:
            self._causal_mask = None

    def forward(self, x, mask=None):       
        # Compute (unmasked) attention matrix
        attn_matrix = torch.bmm(self._Q(x), torch.transpose(self._K(x), 1, 2)) / torch.sqrt(self._dk)

        # If causal, replace upper triangle of attention matrix by -inf
        if self._causal_mask is not None:
            batch_size, length, _ = attn_matrix.shape
            causal_mask = self._causal_mask[:batch_size, :length, :length]
            attn_matrix = causal_mask * attn_matrix + ~causal_mask * -INF

        # When Boolean `mask` is given, drop disallowed attentions (False -> -Inf)
        if mask is not None:
            mask.to(self._device)
            attn_matrix = mask * attn_matrix + ~mask * -INF

        return torch.bmm(torch.softmax(attn_matrix, dim=2), self._V(x))


class MultiHeadSelfAttention(torch.nn.Module):
    """
        Implementation of Multi-Head Self-Attention (Vaswani et al., 2017)
    """
    def __init__(self, in_channels, out_channels, nhead, max_len):
        super().__init__()
        # List of K self-attention heads and reduction layer
        self._heads = torch.nn.ModuleList([SelfAttention(in_channels, out_channels, max_len) for _ in range(nhead)])
        self._linear = torch.nn.Linear(nhead * out_channels, out_channels)

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
    """
        Self-Attention Block with residual connections and LayerNorm
    """
    def __init__(self, d_model, nhead, max_len):
        super().__init__()
        self._attn = MultiHeadSelfAttention(d_model, d_model, nhead=nhead, max_len=max_len)
        self._layer_norm = torch.nn.LayerNorm(d_model)

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, x, mask=None):
        return self._layer_norm(self._attn(x, mask=mask) + x)


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
