import time
import torch

INF = 1e24


class CausalSelfAttention(torch.nn.Module):
    """ 
        Implementation of a Linear Self-Attention layer
        (Vaswani et al., 2017) with Causal Masking
    """
    def __init__(self, d_model, dk=16, maxlen=512):
        super(CausalSelfAttention, self).__init__()
        self._Q = torch.nn.Linear(d_model, dk)
        self._K = torch.nn.Linear(d_model, dk)
        self._V = torch.nn.Linear(d_model, d_model)
        self._inv_dk = torch.sqrt(torch.tensor(dk))

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

        # Precompute causal mask of (batch_size, maxlen, maxlen)
        self._causal_mask = torch.triu(torch.ones(maxlen, maxlen, device=self._device), diagonal=1).bool().unsqueeze(0)

    def forward(self, x, padding_mask):
        # Compute Q/K/V matrices
        q = self._Q(x)
        k = self._K(x)
        v = self._V(x)

        # Mask keys using padding matrix to drop attention to padding timesteps to -INF
        if padding_mask is not None:
            k.masked_fill_(mask=padding_mask, value=-INF)

        # Compute attention matrix QK^T/sqrt(dk)
        attention = torch.bmm(q, torch.transpose(k, 1, 2)) * self._inv_dk

        # Make causal by replacing upper triangle of attn_mat by -inf
        causal_mask = self._causal_mask[:, :x.size(1), :x.size(1)]
        attention.masked_fill_(mask=causal_mask, value=-INF)

        # Complete attention computation
        return torch.bmm(torch.softmax(attention, dim=2), v)


class MultiHeadSelfAttention(torch.nn.Module):
    """
        Implementation of Multi-Head Self-Attention (Vaswani et al., 2017)
    """
    def __init__(self, d_model, nheads=2, dk=16, maxlen=512):
        super().__init__()
        # List of K self-attention heads and reduction layer
        self._heads = torch.nn.ModuleList([CausalSelfAttention(d_model, dk=dk, maxlen=maxlen) for _ in range(nheads)])
        self._linear = torch.nn.Linear(nheads * d_model, d_model)

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, x, padding_mask=None):
        hidden = torch.concat([head(x, padding_mask) for head in self._heads], dim=2)
        return self._linear(hidden)


class SelfAttentionBlock(torch.nn.Module):
    """
        Self-Attention Block with residual connections and LayerNorm
    """
    def __init__(self, d_model, out_channels, nheads=1, dk=16, maxlen=512):
        super(SelfAttentionBlock, self).__init__()
        # If only one head is used, directly use CausalLinearAttention (wee bit faster)
        if nheads < 2:
            self._attn = CausalSelfAttention(d_model, dk=dk, maxlen=maxlen)
        else:
            self._attn = MultiHeadSelfAttention(d_model, nheads=nheads, dk=dk, maxlen=maxlen)

        self._linear = torch.nn.Linear(d_model, out_channels)
        self._layer_norm = torch.nn.LayerNorm(d_model)
        self._leaky_relu = torch.nn.LeakyReLU()

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, x, padding_mask=None, return_last=False):
        h = self._layer_norm(self._attn(x, padding_mask) + x)

        # If return_last, feed only last step through Linear (faster)
        y = self._linear(h[:, -1]) if return_last else self._linear(h)
        return self._leaky_relu(y)


if __name__ == '__main__':
    X = torch.randn((64, 512, 48))  # -> (batch_size, seq_length, num_features)
    print('In:', X.shape)

    attn = SelfAttentionBlock(d_model=48, nheads=1)

    # Sanity check: time forward pass
    def timing_test(model, inputs, N=50):
        time1 = time.time_ns()
        for _ in range(N):
            model(inputs)
        time_elapsed = (time.time_ns() - time1) / (1e9 * N)
        return time_elapsed

    print('Time elapsed:', timing_test(attn, X))
