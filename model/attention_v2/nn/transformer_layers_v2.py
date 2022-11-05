import torch
import torch.nn.functional as F


class TransformerEncoderLayer(torch.nn.Module):
    """
    Simplified Implementation of PyTorch's TransformerEncoderLayer
    with residual connections and layer normalization
    """
    def __init__(self, d_model, layer_norm_eps=1e-5, n_heads=1, d_key=32, d_head=None, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        assert n_heads > 0 and isinstance(n_heads, int)

        # If one head is used, we use simple SA instead
        if n_heads < 2:
            self.self_attn = SelfAttention(d_model, d_key=d_key)
        else:
            self.self_attn = MultiHeadSelfAttention(d_model, n_heads=n_heads, d_key=d_key, d_head=d_head)

        self.feedforward = torch.nn.Linear(d_model, d_model)
        self.layer_norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layer_norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout2 = torch.nn.Dropout(p=dropout)
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x, attn_mask, pairwise_dist):
        # Attention layer + residual connections
        x_attn = self.self_attn(x, attn_mask=attn_mask, pairwise_dist=pairwise_dist)
        z = self.dropout1(self.layer_norm1(x + x_attn))

        # Feed forward layer + residual connections
        z_ff = self.leaky_relu(self.feedforward(z))
        return self.dropout2(self.layer_norm2(z + z_ff))


class MultiHeadSelfAttention(torch.nn.Module):
    """
    Implementation of Multi-Head Self-Attention (Vaswani et al., 2017)
    """
    def __init__(self, d_model, n_heads, d_key, d_head=None):
        super(MultiHeadSelfAttention, self).__init__()
        assert n_heads > 0

        # Allow output dimensions different from d_model for each head
        d_head = d_head if d_head is not None else d_model

        self._heads = torch.nn.ModuleList()
        for _ in range(n_heads):
            self._heads.append(SelfAttention(d_model, d_key=d_key, d_val=d_head))

        self._feedforward = torch.nn.Linear(d_head * n_heads, d_model)

    def forward(self, x, attn_mask, pairwise_dist):
        """ Forward pass """
        y = [h(x, attn_mask=attn_mask, pairwise_dist=pairwise_dist) for h in self._heads]
        return self._feedforward(torch.concat(y, dim=2))


class SelfAttention(torch.nn.Module):
    """
    Implementation of a Simple Self-Attention layer
    """
    def __init__(self, d_model, d_key=32, d_val=None):
        super(SelfAttention, self).__init__()
        # Allow output dimensions different from d_model
        d_val = d_val if d_val is not None else d_model

        self._Q = torch.nn.Linear(d_model, d_key, bias=False)
        self._K = torch.nn.Linear(d_model, d_key, bias=False)
        self._V = torch.nn.Linear(d_model, d_val, bias=False)
        self._omega = torch.nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _softmax(x, dim):
        """ Computes softmax while taking care of -inf entries """
        z = torch.exp(x)
        return z / (torch.sum(z, dim=dim, keepdim=True) + 1)

    def forward(self, x, attn_mask, pairwise_dist):
        """ Forward pass """
        # We normalize the keys and queries as this helps desaturate softmax
        # See: https://aclanthology.org/2020.findings-emnlp.379.pdf
        q = F.normalize(self._Q(x), p=2.0, dim=2)
        k = F.normalize(self._K(x), p=2.0, dim=2)

        self_attn = torch.matmul(q, torch.transpose(k, 1, 2))  # a_ij ~ [-1, 1]

        # Attenuate long-distances using exponential decay parameterized by omega
        r = torch.exp(-self._omega * pairwise_dist) - 1  # r_ij ~ (-1, 0]
        self_attn = self_attn + r.detach()

        # `attn_mask` can be used to mask out future positions and padding positions
        if attn_mask is not None:
            self_attn = self_attn.masked_fill(mask=attn_mask, value=-float('inf'))

        return torch.bmm(self._softmax(self_attn, dim=2), self._V(x))
