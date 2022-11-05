import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt


class TransformerEncoderLayer(torch.nn.Module):
    """
    Simplified Implementation of PyTorch's TransformerEncoderLayer
    with residual connections and layer normalization
    """
    def __init__(self, d_model, layer_norm_eps=1e-5, d_key=32, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = SelfAttention(d_model, d_key=d_key)
        self.feedforward = torch.nn.Linear(d_model, d_model)
        self.layer_norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.layer_norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout2 = torch.nn.Dropout(p=dropout)
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x, attn_mask):
        z = self.layer_norm1(x + self.self_attn(x, attn_mask=attn_mask))
        z = self.dropout1(z)
        y = self.layer_norm2(z + self.leaky_relu(self.feedforward(z)))
        return self.dropout2(y)


class SelfAttention(torch.nn.Module):
    """
    Implementation of a Simple Self-Attention layer
    """
    def __init__(self, d_model, d_key=32):
        super(SelfAttention, self).__init__()
        self._Q = torch.nn.Linear(d_model, d_key, bias=False)
        self._K = torch.nn.Linear(d_model, d_key, bias=False)
        self._V = torch.nn.Linear(d_model, d_model, bias=False)

        dk = torch.sqrt(torch.tensor(d_key))
        self._dk = torch.nn.Parameter(dk, requires_grad=False)

    @staticmethod
    def _softmax(x, dim):
        """ Computes softmax while taking care of -inf entries """
        z = torch.exp(x)
        return z / (torch.sum(z, dim=dim, keepdim=True) + 1)

    def forward(self, x, attn_mask, return_attn=False):
        if torch.any(torch.isnan(x)):
            raise Exception('NaN encountered in SelfAttention.forward')

        # We normalize the keys and queries as this helps desaturate softmax
        # See: https://aclanthology.org/2020.findings-emnlp.379.pdf
        q = F.normalize(self._Q(x), p=2.0, dim=2)
        k = F.normalize(self._K(x), p=2.0, dim=2)
        v = self._V(x)

        self_attn = torch.matmul(q, torch.transpose(k, 1, 2)) / self._dk

        # `attn_mask` can be used to mask out future positions and padding positions
        if attn_mask is not None:
            self_attn = self_attn.masked_fill(mask=attn_mask, value=-float('inf'))

        if return_attn:
            return self_attn

        return torch.bmm(self._softmax(self_attn, dim=2), v)
