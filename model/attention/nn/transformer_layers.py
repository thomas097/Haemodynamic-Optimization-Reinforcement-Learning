import torch
import torch.nn.functional as F
import math


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_key, conv_size=1):
        """
        Transformer encoder block with residual connections and layer normalization
        :param d_model:   Latent dimensions of the encoder
        :param n_heads:   Number of self-attention heads
        :param d_key:     Dimensions of queries/keys in self-attention computation
        :param conv:      Number of time steps to take into account when computing convolution
        """
        super(TransformerEncoderLayer, self).__init__()
        assert n_heads > 0 and d_model > 0 and isinstance(n_heads, int)
        self._conv_size= conv_size

        # if one head is used, we use simple SA instead
        if n_heads < 2:
            self.self_attn = SelfAttention(d_model, d_key=d_key)
        else:
            self.self_attn = MultiHeadSelfAttention(d_model, n_heads=n_heads, d_key=d_key)

        # we replace the position-wise feedforward layer by a 1D convolutional layer
        self.feedforward = torch.nn.Conv1d(d_model, d_model, kernel_size=conv_size, padding=0)
        self.layer_norm1 = torch.nn.LayerNorm(d_model)
        self.layer_norm2 = torch.nn.LayerNorm(d_model)
        self.leaky_relu = torch.nn.LeakyReLU()

    def _causal_padding(self, x):
        """ Adds causal padding to input sequence to ensure causal feed forward pass """
        return F.pad(x, pad=(0, 0, self._conv_size - 1, 0), mode='constant', value=0)

    def forward(self, x, src_mask, rel_dists):
        """
        Forward pass through encoder block
        :param x:           Tensor of shape (batch_size, num_timesteps, d_model)
        :param src_mask:    Attention mask of shape (batch_size, num_timesteps, num_timesteps)
        :param rel_dists:   Distance tensor of shape (batch_size, num_timesteps, num_timesteps)
        :return:            Tensor of shape (batch_size, num_timesteps, d_model)
        """
        # self-attention + residual connections
        x_attn = self.self_attn(x, src_mask=src_mask, rel_dists=rel_dists)
        z = self.layer_norm1(x + x_attn)

        # feed forward using convolution + residual connections
        z_padded = self._causal_padding(z).permute(0, 2, 1)
        z_ff = self.feedforward(z_padded).permute(0, 2, 1)
        return self.layer_norm2(z + self.leaky_relu(z_ff))


class MultiHeadSelfAttention(torch.nn.Module):
    def __init__(self, d_model, n_heads, d_key):
        """ Implementation of Multi-Head Self-Attention (Vaswani et al., 2017)
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
        """ Self-Attention Module (Vaswani et al., 2017)
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

        sqrt_dk = torch.tensor(1 / math.sqrt(d_key))
        self.register_buffer('_sqrt_dk', sqrt_dk, persistent=True)

    def forward(self, x, src_mask, rel_dists):
        """ Forward pass through Self-Attention layer
        :param x:           Tensor of shape (batch_size, n_timesteps, d_model)
        :param src_mask:    Attention mask of shape (batch_size, n_timesteps, n_timesteps)
        :param rel_dists:   Distance tensor of shape (batch_size, n_timesteps, n_timesteps)
        :return:            Tensor of shape (batch_size, n_timesteps, d_model)
        """
        q = self._Q(x)
        k = self._K(x)
        attn_logits = torch.matmul(q, torch.transpose(k, 1, 2)) * self._sqrt_dk

        # relative positional encoding
        attn_logits = attn_logits - self._pos_encoding(rel_dists)

        if src_mask is not None:
            attn_logits = attn_logits.masked_fill(mask=src_mask, value=-1e9)

        # bugfix: if rows are all -inf as a result of combined padding/causal masking,
        # softmax will underflow causing divide-by-zero, so we set all entries to 1.0 to prevent this
        with torch.no_grad():
            fix_mask = torch.all(attn_logits < -1e7, dim=2, keepdim=True)
        attn_logits = attn_logits.masked_fill(mask=fix_mask, value=1.0)

        return torch.bmm(torch.softmax(attn_logits, dim=2), self._V(x))
