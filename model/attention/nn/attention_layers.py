import time
import torch


INF = 1e24


class SelfAttention(torch.nn.Module):
    """ 
        Implementation of a Self-Attention layer (Vaswani et al., 2017)
        with causal attention masking
    """
    def __init__(self, d_model, max_len, dk=16, is_causal=True):
        super().__init__()
        self._Q = torch.nn.Linear(d_model, dk, bias=True)  # Speed up with Keys = Queries!
        self._V = torch.nn.Linear(d_model, d_model, bias=True)
        self._dk = 1 / torch.sqrt(torch.tensor(dk))

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

        # Precompute causal mask of (batch_size, max_len, max_len)
        if is_causal:
            mask = -INF * torch.triu(torch.ones(max_len, max_len), diagonal=1)
            self._causal_mask = mask.unsqueeze(0).to(self._device)
        else:
            self._causal_mask = None

    def forward(self, x, mask=None):       
        # Compute attention matrix
        queries = self._Q(x)
        attn_matrix = torch.bmm(queries, torch.transpose(queries, 1, 2)) * self._dk

        # If causal; replace upper triangle of attn_matrix by -inf
        if self._causal_mask is not None:
            attn_matrix = attn_matrix + self._causal_mask[:, :x.size(1), :x.size(1)]

        # When `mask` is given, drop disallowed attentions (True -> -Inf)
        if mask is not None:
            attn_matrix = attn_matrix + mask * -INF

        return torch.bmm(torch.softmax(attn_matrix, dim=2), self._V(x))


class MultiHeadSelfAttention(torch.nn.Module):
    """
        Implementation of Multi-Head Self-Attention (Vaswani et al., 2017)
    """
    def __init__(self, d_model, nhead, max_len):
        super().__init__()
        # List of K self-attention heads and reduction layer
        self._heads = torch.nn.ModuleList([SelfAttention(d_model, max_len) for _ in range(nhead)])
        self._linear = torch.nn.Linear(nhead * d_model, d_model)

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, x, mask=None):
        hidden = torch.concat([head(x, mask=mask) for head in self._heads], dim=2)
        return self._linear(hidden)


class SelfAttentionBlock(torch.nn.Module):
    """
        Self-Attention Block with residual connections and LayerNorm
    """
    def __init__(self, d_model, nhead, max_len=512):
        super(SelfAttentionBlock, self).__init__()
        # If only one head is used, simply use SelfAttention (wee bit faster)
        if nhead < 2:
            self._attn = SelfAttention(d_model, max_len=max_len)
        else:
            self._attn = MultiHeadSelfAttention(d_model, nhead=nhead, max_len=max_len)

        self._layer_norm = torch.nn.LayerNorm(d_model)

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, x, mask=None):
        return self._layer_norm(self._attn(x, mask=mask) + x)


if __name__ == '__main__':
    X = torch.randn((64, 512, 48))  # -> (batch_size, seq_length, num_features)
    print('In:', X.shape)

    attn = SelfAttentionBlock(d_model=48, nhead=1, max_len=512)

    # Sanity check: time forward pass
    def timing_test(model, inputs, N=50):
        time1 = time.time_ns()
        for _ in range(N):
            model(inputs)
        time_elapsed = (time.time_ns() - time1) / (1e9 * N)
        return time_elapsed

    print('Time elapsed:', timing_test(attn, X))
