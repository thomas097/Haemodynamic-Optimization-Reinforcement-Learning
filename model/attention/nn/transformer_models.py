import torch
import torch.nn.functional as F
from transformer_layers import TransformerEncoderLayer


class Transformer(torch.nn.Module):
    def __init__(self, vocab_size, out_channels, d_model=64, d_key=32, n_blocks=2, n_heads=2, padding_value=0,
                 mask_features=True, causal=True):
        """
        Transformer (Vaswani et al., 2017) with Causal Self-Attention

        :param vocab_size:       Num modalities
        :param out_channels:     Num output channels
        :param d_model:          Latent dimensions of the model (default: 64)
        :param d_key:            Latent dimensions of queries/keys (default: 32)
        :param n_blocks:         Num encoder blocks (default: 2)
        :param n_heads:          Num self-attention heads per block (default: 2)
        :param padding_value:    Vocab entry reserved for padding (default: 0)
        :param causal:           Whether to mask future positions in attention calculation
        :param mask_features:    Whether to enforce a strict within-feature attention scheme whereby
                                 observations may only attend to observations of the same feature
        """
        super(Transformer, self).__init__()
        self.config = locals()
        self._padding_value = padding_value
        self._vocab_size = vocab_size
        self._causal = causal
        self._mask_features = mask_features

        # Input encoding network
        self._input_encoding = torch.nn.Sequential(
            torch.nn.Linear(vocab_size + 1, 96),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(96, d_model)
        )

        # Transformer blocks + fusion block + FF layer
        self._encoder_layers = torch.nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads=n_heads, d_key=d_key) for _ in range(n_blocks)]
        )
        self._fusion = TransformerEncoderLayer(d_model, n_heads=n_heads, d_key=d_key)
        self._linear = torch.nn.Linear(d_model, out_channels)
        self._initialize()

    def _initialize(self):
        # Initialize model with Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    @staticmethod
    def _distance_matrix(t):
        """ Computes a matrix of pairwise distances between observations.
        Used for relative positional encoding (RPE)
        :param t:  Tensor of timesteps for each measurement
        :returns:  FloatTensor of shape (batch_size, 1, num_timesteps, num_timesteps)
        """
        return torch.abs(t.permute(0, 2, 1) - t).unsqueeze(1).detach()

    @staticmethod
    def _feature_mask(f):
        """ Computes mask to mask out attention across modalities / features
        :param m:  Tensor of feature ids for each measurement
        :returns:  Boolean Tensor of shape (batch_size, num_timesteps, num_timesteps)
        """
        return ((f.permute(0, 2, 1) - f) != 0).detach()

    @staticmethod
    def _causal_mask(t):
        """ Computes causal mask from timesteps in x_pos
        :param t:  Tensor of time steps for each measurement
        :return:   Boolean mask of shape (batch_size, num_timesteps, num_timesteps)
        """
        return (t.permute(0, 2, 1) - t) > 0

    def _padding_mask(self, x):
        """ Constructs padding mask from all zero entries in x
        :param x:  Input tensor (see forward())
        :return:   Boolean mask of shape (batch_size, 1, num_timesteps)
        """
        is_padding = torch.all(x == self._padding_value, dim=2, keepdim=True)
        return torch.transpose(is_padding, 1, 2)

    def _encode_input(self, x_type, x_val):
        """ Encodes sequence of (measurement_type, value) pairs into a latent vector
        :param x_type:  Tensor of shape (batch_size, num_timesteps, 1) containing integer measurement types
        :param x_val:   Tensor of shape (batch_size, num_timesteps, 1) containing real measurement values
        :return:        Tensor of shape (batch_size, num_timesteps, d_model)
        """
        # One-hot encode x_type (i.e. measurement types) and jointly model with x_val
        x_type = F.one_hot(x_type[:, :, 0].long(), num_classes=self._vocab_size)
        z = torch.concat([x_type, x_val], dim=2)
        return self._input_encoding(z)

    def forward(self, x, return_last=True):
        """ Forward pass through transformer
        :param x:            Tensor of shape (batch_size, num_timesteps, 3) where the features
                             correspond to measurement_type, value and timestep respectively
        :param return_last:  Whether to return the output of the final timestep only
        :return:             Tensor of shape (batch_size, num_timesteps, out_channels)
                             or (batch_size, out_channels) when `return_last=True`
        """
        f, v, t = torch.split(x, 1, dim=2)

        # Create combined causal/padding/modality mask
        src_mask = self._padding_mask(x)
        if self._causal:
            causal_mask = self._causal_mask(t)
            src_mask = torch.logical_or(src_mask, causal_mask).detach()

        if self._mask_features:
            feature_mask = self._feature_mask(f)
            src_mask = torch.logical_or(src_mask, feature_mask).detach()

        # Compute RPE distances
        rel_dist = self._distance_matrix(t).detach()

        y = self._encode_input(f, v)
        for layer in self._encoder_layers:
            y = layer(y, src_mask=src_mask, rel_dists=rel_dist)

        # only mask padding, but allow 'fusion' of channels
        y = self._fusion(y, src_mask=self._padding_mask(x), rel_dists=rel_dist)

        return self._linear(y[:, -1] if return_last else y)
