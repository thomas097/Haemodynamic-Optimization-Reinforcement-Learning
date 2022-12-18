import torch
import torch.nn.functional as F
from transformer_layers import TransformerEncoderLayer


class Transformer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, d_model=64, d_key=24, n_blocks=2, n_heads=2, padding_value=0,
                 max_steps=32, causal=True, conv_size=1):
        """ Transformer (Vaswani et al., 2017) with Causal Self-Attention for time series
        :param in_channels:      Number of input channels
        :param out_channels:     Number of output channels
        :param d_model:          Latent dimensions of the model (default: 64)
        :param d_key:            Latent dimensions of queries/keys (default: 32)
        :param n_blocks:         Num encoder blocks (default: 2)
        :param n_heads:          Num self-attention heads per block (default: 2)
        :param padding_value:    Vocab entry reserved for padding (default: 0)
        :param max_steps:        Maximum number of relative time steps to consider in attention computation
        :param causal:           Whether to mask future positions in attention calculation
        :param conv_size:        Number of time steps to consider in the convolution replacing the
                                 feed forward layer (default: 1; equivalent to simple feed forward)
        """
        super(Transformer, self).__init__()
        self.config = locals()
        self._padding_value = padding_value
        self._max_steps = max_steps
        self._causal = causal
        self._conv_size = conv_size

        # Input encoding network using small convolution
        self._input_encoding = torch.nn.Conv1d(in_channels, d_model, kernel_size=conv_size, padding=0)

        # Transformer blocks + fusion block + FF layer
        self._encoder_layers = torch.nn.ModuleList(
            [TransformerEncoderLayer(d_model, n_heads=n_heads, d_key=d_key, conv_size=conv_size) for _ in range(n_blocks)]
        )
        self._linear = torch.nn.Linear(d_model, out_channels)
        self._initialize()

    def _initialize(self):
        # Initialize model with Xavier initialization
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def _causal_padding(self, x):
        """ Adds causal padding to input sequence to ensure causal feed forward pass """
        return F.pad(x, pad=(self._conv_size - 1, 0), mode='constant', value=0)

    @staticmethod
    def _causal_mask(x):
        """ Computes an upper-triangular matrix to mask out future positions
        :param n_timesteps:  Number of time steps in sequence
        :returns:            FloatTensor of shape (batch_size, 1, num_timesteps, num_timesteps)
        """
        n_timesteps = x.size(1)
        tri = torch.triu(torch.ones((1, n_timesteps, n_timesteps)), diagonal=1).bool()
        return tri.detach().to(x.device)

    def _distance_matrix(self, x):
        """ Computes a matrix of signed distances between observations
        for learning relative positional encoding (RPE)
        :param n_timesteps:  Number of time steps in sequence
        :returns:            FloatTensor of shape (batch_size, 1, num_timesteps, num_timesteps)
        """
        # create signed distance matrix
        n_timesteps = x.size(1)
        timesteps = torch.arange(n_timesteps).unsqueeze(0).unsqueeze(2).float().to(x.device)
        dists = (timesteps.permute(0, 2, 1) - timesteps).unsqueeze(1)

        # limit distances between -max_steps and max_steps to (-1, 1)
        dists = torch.clamp(dists, min=-self._max_steps, max=self._max_steps) / self._max_steps
        return dists.detach()

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
        # truncate sequence to max_steps
        x = x[:, -self._max_steps:]

        # combined causal/padding mask
        src_mask = self._padding_mask(x)
        if self._causal:
            src_mask = torch.logical_or(src_mask, self._causal_mask(x)).to(x.device)

        # distances for RPE
        rel_dist = self._distance_matrix(x).to(x.device)

        # encode input using small convolution
        x_padded = self._causal_padding(x.permute(0, 2, 1))
        y = self._input_encoding(x_padded).permute(0, 2, 1)

        # forward pass
        for layer in self._encoder_layers:
            y = layer(y, src_mask=src_mask, rel_dists=rel_dist)

        return self._linear(y[:, -1] if return_last else y)


if __name__ == '__main__':
    x = torch.randn(8, 72, 48)

    model = Transformer(in_channels=48, out_channels=32)
    print('out:',  model(x).shape)