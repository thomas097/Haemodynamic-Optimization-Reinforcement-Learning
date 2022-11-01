import torch
import torch.nn.functional as F


class TransformerEncoderLayer(torch.nn.Module):
    """
    Simplified Implementation of PyTorch's TransformerEncoderLayer
    """
    def __init__(self, d_model, layer_norm_eps=1e-5, d_key=32):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = SelfAttention(d_model, d_key=d_key)
        self.feedforward = torch.nn.Linear(d_model, d_model)
        self.layer_norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layer_norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)

    def forward(self, x, attn_mask, key_padding_mask):
        x_attn = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        z = self.layer_norm1(x + x_attn)
        z_ff = self.feedforward(z)
        y = self.layer_norm2(z + z_ff)
        return y


class SelfAttention(torch.nn.Module):
    """
        Implementation of a Simple Self-Attention layer
    """
    def __init__(self, d_model, d_key=32):
        super(SelfAttention, self).__init__()
        self._Q = torch.nn.Linear(d_model, d_key, bias=False)
        self._K = torch.nn.Linear(d_model, d_key, bias=False)
        self._V = torch.nn.Linear(d_model, d_model, bias=False)
        self._inv_dk = 1 / torch.sqrt(torch.tensor(d_key))
        self._ninf = -1e24

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, x, attn_mask, key_padding_mask):
        q = self._Q(x)
        k = self._K(x)
        v = self._V(x)

        # `key_padding_mask` can be used to mask out padding
        if key_padding_mask is not None:
            k.masked_fill_(mask=key_padding_mask, value=self._ninf)

        self_attn = torch.bmm(q, torch.transpose(k, 1, 2)) * self._inv_dk

        # `attn_mask` can be used to mask out future positions
        if attn_mask is not None:
            self_attn.masked_fill_(mask=attn_mask, value=self._ninf)

        return torch.bmm(torch.softmax(self_attn, dim=2), v)


class SelfAttentionHead(torch.nn.Module):
    def __init__(self, d_model, out_channels, n_queries=8, d_key=32):
        super(SelfAttentionHead, self).__init__()
        if out_channels % n_queries != 0:
            raise Exception('out_channels must be divisible by n_queries (%d)' % n_queries)

        self._q = torch.nn.Parameter(torch.rand(1, n_queries, d_key))  # Learn queries!
        self._K = torch.nn.Linear(d_model, d_key)
        self._V = torch.nn.Linear(d_model, out_channels // n_queries)

        self._inv_dk = 1 / torch.sqrt(torch.tensor(d_key))
        self._d_out = out_channels
        self._n_queries = n_queries
        self._ninf = -1e24

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, x, key_padding_mask=None):
        # `key_padding_mask` can be used to mask out padding
        k = self._K(x)
        if key_padding_mask is not None:
            k.masked_fill_(mask=key_padding_mask, value=self._ninf)

        # Self-attention computation
        attn = torch.matmul(self._q, torch.transpose(k, 1, 2)) * self._inv_dk
        y = torch.bmm(torch.softmax(attn, dim=2), self._V(x))

        # Reshape to batch_size x out_channels
        return y.reshape(-1, self._d_out)
