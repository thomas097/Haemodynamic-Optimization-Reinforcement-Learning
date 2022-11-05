import torch
import torch.nn.functional as F
from transformer_layers_v2 import TransformerEncoderLayer


class CausalTransformer(torch.nn.Module):
    """ Implementation of a Transformer (Vaswani et al., 2017)
        with causal self-attention_v2
    """
    def __init__(self, vocab_size, out_channels, d_model=64, d_key=32, n_blocks=3, n_heads=2,
                 d_head=None, truncate=0, padding_value=0):
        super(CausalTransformer, self).__init__()
        self.config = locals()

        self._input_encoding = torch.nn.Sequential(
            torch.nn.Linear(vocab_size + 1, d_model),  # measurement type + value
            torch.nn.LeakyReLU(),
            torch.nn.Linear(d_model, d_model)
        )

        self._blocks = torch.nn.ModuleList()
        for _ in range(n_blocks):
            block = TransformerEncoderLayer(d_model=d_model, n_heads=n_heads, d_key=d_key, d_head=d_head)
            self._blocks.append(block)
        self._feedforward = torch.nn.Linear(d_model, out_channels)
        self._initialize()

        self._vocab_size = vocab_size
        self._padding_value = padding_value
        self._truncate = truncate

    def _initialize(self):
        """ Initializes weights using Xavier initialization """
        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    @staticmethod
    def _distance_matrix(x):
        """ Computes a matrix of absolute pairwise distances """
        t = x[:, :, 2].unsqueeze(2)  # Positions
        return torch.abs(t.permute(0, 2, 1) - t)

    @staticmethod
    def _causal_mask(t):
        """ Computes an upper triangular mask to mask out future positions """
        t = t[:, :, 2].unsqueeze(2)  # Positions
        return (t.permute(0, 2, 1) - t) > 0

    def _padding_mask(self, x):
        """ Computes block mask to mask out padding tokens """
        mask = x[:, :, 0].unsqueeze(2)  # Type
        return torch.transpose(mask == self._padding_value, 1, 2)

    def _pack(self, x):
        """ Packs observation (type, value) into a one-hot encoded vector """
        x_val = x[:, :, 1].unsqueeze(2)
        x_type = F.one_hot(x[:, :, 0].long(), num_classes=self._vocab_size)
        return torch.concat([x_val, x_type], dim=2)

    def forward(self, x, return_last=False):
        """ Forward pass through network """
        # Truncate long sequences yo some maximum length
        x = x[:, -self._truncate:].float()

        # Mask zero-padding & future positions using Boolean masks
        attn_mask = torch.logical_or(self._padding_mask(x),
                                     self._causal_mask(x)).detach()
        # Relative distance matrix
        pairwise_dist = self._distance_matrix(x).detach()

        # Forward pass
        y = self._input_encoding(self._pack(x).detach())
        for block in self._blocks:
            y = block(y, attn_mask=attn_mask, pairwise_dist=pairwise_dist)
        return self._feedforward(y[:, -1] if return_last else y)


if __name__ == '__main__':
    # Toy dataset of (measurement type, value, charttime) tuples
    X = torch.randn(32, 100, 3)
    X[:, :, 0] = torch.randint(1, 10, size=(32, 100)).float()
    X[:, :, 2] = torch.cumsum((torch.rand(32, 100) > 0.5).float(), dim=1)
    X[:, :25] = 0  # zero-padding

    model = CausalTransformer(vocab_size=10, out_channels=96, d_model=64, n_heads=3, d_key=32)

    model(X)