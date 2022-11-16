import torch
from transformer_layers import TransformerEncoderLayer


class Transformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, d_model=64, d_key=32, n_blocks=2, n_heads=2, padding_value=0, causal=True):
        """ Transformer (Vaswani et al., 2017) with Causal Self-Attention
        :param in_channels:      Number of input channels
        :param out_channels:     Number of output channels
        :param d_model:          Latent dimensions of the model (default: 64)
        :param d_key:            Latent dimensions of queries/keys (default: 32)
        :param n_blocks:         Num encoder blocks (default: 2)
        :param n_heads:          Num self-attention heads per block (default: 2)
        :param padding_value:    Vocab entry reserved for padding (default: 0)
        :param causal:           Whether to mask future positions in attention calculation
        """
        super(Transformer, self).__init__()
        self.config = locals()
        self._padding_value = padding_value
        self._causal = causal

        # Input encoding network
        self._input_encoding = torch.nn.Linear(in_channels, d_model)

        # Transformer blocks + fusion block + FF layer
        self._encoder_layers = torch.nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads=n_heads, d_key=d_key) for _ in range(n_blocks)]
        )
        self._linear = torch.nn.Linear(d_model, out_channels)
        self._initialize()

    def _initialize(self):
        # Initialize model with Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    @staticmethod
    def _causal_mask(x):
        tri = torch.triu(torch.ones((1, x.size(1), x.size(1))), diagonal=1).bool()
        return tri.to(x.device).detach()

    @staticmethod
    def _distance_matrix(x):
        t = torch.arange(x.size(1)).unsqueeze(0).unsqueeze(2).float()
        return torch.cdist(t, t).unsqueeze(0).to(x.device).detach()

    def _padding_mask(self, x):
        """ Constructs padding mask from all zero entries in x
        :param x:  Input tensor (see forward())
        :return:   Boolean mask of shape (batch_size, 1, num_timesteps)
        """
        is_padding = torch.all(x == self._padding_value, dim=2, keepdim=True)
        return torch.transpose(is_padding, 1, 2)

    def forward(self, x, return_last=True):
        """ Forward pass through transformer
        :param x:            Tensor of shape (batch_size, n_timesteps, n_features)
        :param return_last:  Whether to return the output of the final timestep only
        :return:             Tensor of shape (batch_size, num_timesteps, out_channels)
                             or (batch_size, out_channels) when `return_last=True`
        """
        # Create combined causal/padding mask
        src_mask = self._padding_mask(x)
        if self._causal:
            src_mask = torch.logical_or(src_mask, self._causal_mask(x)).detach()

        # Compute RPE distances
        rel_dist = self._distance_matrix(x)

        y = self._input_encoding(x)
        for layer in self._encoder_layers:
            y = layer(y, src_mask=src_mask, rel_dists=rel_dist)

        return self._linear(y[:, -1] if return_last else y)
