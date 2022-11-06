import torch
import torch.nn.functional as F


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

    def forward(self, x, src_mask, distances):
        """
        Forward pass through encoder block

        :param x:           Input tensor of shape (batch_size, num_timesteps, d_model)
        :param src_mask:    Attention mask of shape (batch_size, num_timesteps, num_timesteps)
        :param distances:   Distance tensor of shape (batch_size, num_timesteps, num_timesteps)
        :return:            Tensor of shape (batch_size, num_timesteps, d_model)
        """
        # Self-attention + residual connections
        x_attn = self.self_attn(x, src_mask=src_mask, distances=distances)
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

    def forward(self, x, src_mask, distances):
        """
        Forward pass through Self-Attention layer

        :param x:           Input tensor of shape (batch_size, num_timesteps, d_model)
        :param src_mask:    Attention mask of shape (batch_size, num_timesteps, num_timesteps)
        :param distances:   Distance tensor of shape (batch_size, num_timesteps, num_timesteps)
        :return:            Tensor of shape (batch_size, num_timesteps, d_model)
        """
        y = torch.concat([h(x, src_mask, distances) for h in self._heads], dim=2)
        return self._feedforward(y)


class SelfAttention(torch.nn.Module):
    def __init__(self, d_model, d_key=32, d_val=None):
        """
        Self-Attention Module (Vaswani et al., 2017)

        :param d_model:  Latent dimensions of the encoder
        :param d_key:    Dimensions of queries/keys in self-attention computation
        :param d_val:    Dimensions of values in self-attention computation (optional).
                         If `d_val=None`, it is assumed d_val == d_model.
        """
        super(SelfAttention, self).__init__()
        # Allow output dimensions different from d_model
        d_val = d_val if d_val is not None else d_model

        self._Q = torch.nn.Linear(d_model, d_key, bias=False)
        self._K = torch.nn.Linear(d_model, d_key, bias=False)
        self._V = torch.nn.Linear(d_model, d_val, bias=False)
        self._p0 = torch.nn.Parameter(torch.tensor(1.0))
        self._p1 = torch.nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def _softmax(x):
        """
        Numerically stable implementation of softmax

        :param x:  Attention matrix of shape (batch_size, num_timesteps, num_timesteps)
        :return:   Normalized attention matrix
        """
        z = torch.exp(x)
        return z / (torch.sum(z, dim=2, keepdim=True) + 1)

    def forward(self, x, src_mask, distances):
        """
        Forward pass through Self-Attention layer

        :param x:           Input tensor of shape (batch_size, num_timesteps, d_model)
        :param src_mask:    Attention mask of shape (batch_size, num_timesteps, num_timesteps)
        :param distances:   Distance tensor of shape (batch_size, num_timesteps, num_timesteps)
        :return:            Tensor of shape (batch_size, num_timesteps, d_model)
        """
        # We normalize keys and queries as this helps bound the attention
        # coefficients and desaturate the softmax
        # See: https://aclanthology.org/2020.findings-emnlp.379.pdf
        q = F.normalize(self._Q(x), p=2.0, dim=2)
        k = F.normalize(self._K(x), p=2.0, dim=2)
        attn = torch.matmul(q, torch.transpose(k, 1, 2))

        # Relative positional encoding using parameterized decay function
        r = self._p1 * (torch.exp(-self._p0 * distances) - 1)  # r_ij ~ (-p1, 0]
        attn = attn + r

        # `src_mask` can be used to mask out future positions and padding
        # positions in attention matrix
        if src_mask is not None:
            attn = attn.masked_fill(mask=src_mask, value=-float('inf'))

        return torch.bmm(self._softmax(attn), self._V(x))
