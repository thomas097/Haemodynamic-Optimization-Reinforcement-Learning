import torch
import torch.nn.functional as F
from transformer_layers import TransformerEncoderLayer


class CausalTransformer(torch.nn.Module):
    def __init__(self, vocab_size, out_channels, d_model=64, d_key=32, n_blocks=2, n_heads=2, padding_value=0):
        """
        Transformer (Vaswani et al., 2017) with Causal Self-Attention

        :param vocab_size:     Number of measurement types in dataset
        :param out_channels:   Number of output channels
        :param d_model:        Latent dimensions of the model (default: 64)
        :param d_key:          Dimensions of queries/keys in self-attention computation (default: 32)
        :param n_blocks:       Number of encoder blocks (default: 2)
        :param n_heads:        Number of attention heads per block (default: 2)
        :param padding_value:  Entry in vocab reserved for padding (default: 0)
        """
        super(CausalTransformer, self).__init__()
        self.config = locals()
        self._padding_value = padding_value
        self._vocab_size = vocab_size

        # Input encoding network
        self._input_encoding = torch.nn.Sequential(
            torch.nn.Linear(vocab_size + 1, 96),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(96, d_model)
        )

        # Transformer blocks + FF layer
        self._encoder_layers = torch.nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads, d_key) for _ in range(n_blocks)]
        )
        self._linear = torch.nn.Linear(d_model, out_channels)
        self._initialize()

    def _initialize(self):
        # Initialize model with Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    @staticmethod
    def _distance_matrix(t):
        """
        Computes relative distances of timesteps in x_pos for
        relative positional encoding (RPE)

        :return: Distance tensor of shape (batch_size, num_timesteps, num_timesteps)
        """
        return torch.abs(t.permute(0, 2, 1) - t).detach()

    @staticmethod
    def _causal_mask(t):
        """
        Computes causal mask from timesteps in x_pos

        :return: Causal mask of shape (batch_size, num_timesteps, num_timesteps)
        """
        return (t.permute(0, 2, 1) - t) > 0

    def _padding_mask(self, t):
        """
        Constructs padding mask from ids in x_type

        :return: Padding mask of shape (batch_size, num_timesteps, num_timesteps)
        """
        return torch.transpose(t == self._padding_value, 1, 2)

    def _encode_input(self, x_type, x_val):
        """
        Encodes sequence of (measurement_type, value) pairs into a latent vector

        :param x_type:  Input tensor of shape (batch_size, num_timesteps, 1)
        :param x_val:   Input tensor of shape (batch_size, num_timesteps, 1)
        :return:        Tensor of shape (batch_size, num_timesteps, d_model)
        """
        # One-hot encode x_type (i.e. measurement types) and jointly model with x_val
        x_type = F.one_hot(x_type[:, :, 0].long(), num_classes=self._vocab_size)
        z = torch.concat([x_type, x_val], dim=2)
        return self._input_encoding(z)

    def forward(self, x, return_last=True):
        """
        Forward pass through transformer

        :param x:            Tensor of shape (batch_size, num_timesteps, 3) where the features
                             correspond to measurement_type, value and timestep respectively
        :param return_last:  Whether to return the output of the final timestep only
        :return:             Tensor of shape (batch_size, num_timesteps, out_channels)
                             or (batch_size, out_channels) when `return_last=True`
        """
        x_type, x_val, x_pos = torch.split(x, 1, dim=2)

        # Create combined causal/padding mask
        padding_mask = self._padding_mask(x_type)
        causal_mask = self._causal_mask(x_pos)
        src_mask = torch.logical_or(padding_mask, causal_mask).detach()

        # Compute RPE distances
        distances = self._distance_matrix(x_pos).detach()

        y = self._encode_input(x_type, x_val)
        for layer in self._encoder_layers:
            y = layer(y, src_mask=src_mask, distances=distances)
        return self._linear(y[:, -1] if return_last else y)
    