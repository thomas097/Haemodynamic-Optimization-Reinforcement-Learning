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
        self._self_attn = SelfAttention(d_model, d_key=d_key)
        self._feedforward = torch.nn.Linear(d_model, d_model)
        self._layer_norm1 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self._layer_norm2 = torch.nn.LayerNorm(d_model, eps=layer_norm_eps)
        self._dropout1 = torch.nn.Dropout(p=dropout)
        self._dropout2 = torch.nn.Dropout(p=dropout)
        self._leaky_relu = torch.nn.LeakyReLU()
        self._initialize_weights()

    def _initialize_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def forward(self, x, attn_mask):
        z = self._layer_norm1(x + self._self_attn(x, attn_mask=attn_mask))
        z = self._dropout1(z)
        y = self._layer_norm2(z + self._leaky_relu(self._feedforward(z)))
        return self._dropout2(y)


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

    def forward(self, x, attn_mask):
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
            self_attn.masked_fill_(mask=attn_mask, value=-float('inf'))

        return torch.bmm(self._softmax(self_attn, dim=2), v)
