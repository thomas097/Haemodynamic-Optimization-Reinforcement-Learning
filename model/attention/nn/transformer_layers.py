import torch
import torch.nn.functional as F
import math


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_key, dropout=0.1):
        """
        Transformer encoder block with residual connections and layer normalization
        :param d_model:   Latent dimensions of the encoder
        :param n_heads:   Number of self-attention heads
        :param d_key:     Dimensions of queries/keys in self-attention computation
        :param dropout:   Dropout rate (default: 0.1)
        """
        super(TransformerEncoderLayer, self).__init__()
        assert n_heads > 0 and d_model > 0 and isinstance(n_heads, int)

        # If one head is used, we use simple SA instead
        if n_heads < 2:
            self.self_attn = SelfAttention(d_model, d_key=d_key)
        else:
            self.self_attn = MultiHeadSelfAttention(d_model, n_heads=n_heads, d_key=d_key)

        self.feedforward = torch.nn.Linear(d_model, d_model)
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(p=dropout)
        self.dropout2 = torch.nn.Dropout(p=dropout)
        self.leaky_relu = torch.nn.LeakyReLU()

    def forward(self, x, src_mask, rel_dists):
        """
        Forward pass through encoder block
        :param x:           Tensor of shape (batch_size, num_timesteps, d_model)
        :param src_mask:    Attention mask of shape (batch_size, num_timesteps, num_timesteps)
        :param rel_dists:   Distance tensor of shape (batch_size, num_timesteps, num_timesteps)
        :return:            Tensor of shape (batch_size, num_timesteps, d_model)
        """
        # Self-attention + residual connections
        x_attn = self.self_attn(x, src_mask=src_mask, rel_dists=rel_dists)
        z = self.dropout1(self.layer_norm1(x + x_attn))

        # Feed forward + residual connections
        z_ff = self.leaky_relu(self.feedforward(z))
        return self.dropout2(self.layer_norm2(z + z_ff))


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_key):
        """
        Implementation of Multi-Head Self-Attention (Vaswani et al., 2017)
        :param d_model:  Latent dimensions of the encoder
        :param n_heads:  Number of self-attention heads
        :param d_key:    Dimensions of queries/keys in self-attention computation
        """
        super(MultiHeadSelfAttention, self).__init__()
        assert n_heads > 0 and d_model % n_heads == 0

        self._heads = torch.nn.ModuleList()
        for _ in range(n_heads):
            self._heads.append(SelfAttention(d_model, d_key=d_key, d_val=d_model // n_heads))

        self._feedforward = torch.nn.Linear(d_model, d_model)

    def forward(self, x, src_mask, rel_dists):
        """
        Forward pass through Self-Attention layer
        :param x:           Tensor of shape (batch_size, num_timesteps, d_model)
        :param src_mask:    Attention mask of shape (batch_size, num_timesteps, num_timesteps)
        :param rel_dists:   Distance tensor of shape (batch_size, num_timesteps, num_timesteps)
        :return:            Tensor of shape (batch_size, num_timesteps, d_model)
        """
        y = torch.concat([h(x, src_mask=src_mask, rel_dists=rel_dists) for h in self._heads], dim=2)
        return self._feedforward(y)


class RelativePositionalEncoding(torch.nn.Module):
    def __init__(self, hidden_units=16):
        super(RelativePositionalEncoding, self).__init__()
        self._pos_encoder = torch.nn.Sequential(
            torch.nn.Conv2d(1, hidden_units, kernel_size=(1, 1), bias=True),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(hidden_units, 1, kernel_size=(1, 1), bias=True),
        )

    def forward(self, attn_mat):
        # drop channel dimension -> (batch_size, *attn_matrix size)
        return self._pos_encoder(attn_mat)[:, 0]


class SelfAttention(torch.nn.Module):
    def __init__(self, d_model, d_key=32, d_val=None):
        """
        Self-Attention Module (Vaswani et al., 2017)
        :param d_model:  Latent dimensions of the encoder
        :param d_key:    Dimensions of queries/keys in self-attention computation
        :param d_val:    Dimensions of values in self-attention computation (optional).
                         If `d_val=None`, it is assumed d_val == d_model
        """
        super(SelfAttention, self).__init__()
        assert d_model > 0 and d_key > 0
        # Allow output dimensions different from d_model
        d_val = d_val if d_val is not None else d_model

        self._Q = torch.nn.Linear(d_model, d_key, bias=False)
        self._K = torch.nn.Linear(d_model, d_key, bias=False)
        self._V = torch.nn.Linear(d_model, d_val, bias=False)
        self._pos_encoding = RelativePositionalEncoding()

    @staticmethod
    def _softmax(x):
        """ Numerically stable implementation of softmax which takes care
        of divide-by-zero caused by padding tokens
        :param x:  Attention matrix with shape (batch_size, num_timesteps, num_timesteps)
        :return:   Normalized attention matrix
        """
        z = torch.exp(x)
        return z / (torch.sum(z, dim=2, keepdim=True) + 1e-5)

    def forward(self, x, src_mask, rel_dists):
        """ Forward pass through Self-Attention layer
        :param x:           Tensor of shape (batch_size, n_timesteps, d_model)
        :param src_mask:    Attention mask of shape (batch_size, n_timesteps, n_timesteps)
        :param rel_dists:   Distance tensor of shape (batch_size, n_timesteps, n_timesteps)
        :return:            Tensor of shape (batch_size, n_timesteps, d_model)
        """
        q = F.normalize(self._Q(x), p=2.0, dim=2)
        k = F.normalize(self._K(x), p=2.0, dim=2)
        v = self._V(x)
        attn_logits = torch.matmul(q, torch.transpose(k, 1, 2))

        # Relative per-channel positional encoding
        attn_logits = attn_logits - self._pos_encoding(rel_dists)

        # `src_mask` can be used to mask out future positions and padding
        if src_mask is not None:
            attn_logits = attn_logits.masked_fill(mask=src_mask, value=-1e15)

        return torch.bmm(self._softmax(attn_logits), v)
