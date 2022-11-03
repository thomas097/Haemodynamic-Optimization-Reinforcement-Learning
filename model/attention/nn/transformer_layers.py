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
        self.leaky_relu = torch.nn.LeakyReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, attn_mask, key_padding_mask):
        h = self.self_attn(x, attn_mask=attn_mask, key_padding_mask=key_padding_mask)
        z = self.layer_norm1(x + h)
        y = self.leaky_relu(self.feedforward(z))
        return self.layer_norm2(z + y)


class SelfAttention(torch.nn.Module):
    """
        Implementation of a Simple Self-Attention layer
    """
    def __init__(self, d_model, d_key=32):
        super(SelfAttention, self).__init__()
        self._Q = torch.nn.Linear(d_model, d_key, bias=False)
        self._K = torch.nn.Linear(d_model, d_key, bias=False)
        self._V = torch.nn.Linear(d_model, d_model, bias=False)

        # Predefine 1 / sqrt(d_key) and -inf
        inv_dk = 1 / torch.sqrt(torch.tensor(d_key))
        self._inv_dk = torch.nn.Parameter(inv_dk, requires_grad=False)
        self._ninf = -1e24

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
