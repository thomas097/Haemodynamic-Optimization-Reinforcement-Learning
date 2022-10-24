import math
import torch


class PositionalEncoding(torch.nn.Module):
    """ Implementation of the Positional Encoding used by (Vaswani et al., 2017)
        For details, please refer to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, batch_size, d_model):
        super(PositionalEncoding, self).__init__()
        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

        # Precompute div term
        i = torch.arange(0, d_model, 2).unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1)  # Expand to batch size!
        self._div_term = torch.exp(i * (-math.log(10000.0) / d_model)).to(self._device)
        self._d_model = d_model

    def forward(self, t):
        """ Converts a sequence of time steps into a sequence of positional embeddings

            t:       Tensor of real-valued time steps of shape (batch_size, num_steps)

            returns: Embedding Tensor of shape (batch_size, num_steps, embedding_dim)
        """
        position = t.to(self._device).unsqueeze(2)
        out = torch.zeros(*t.shape, self._d_model)
        outer = torch.bmm(position, self._div_term)
        out[:, :, 0::2] = torch.sin(outer)
        out[:, :, 1::2] = torch.cos(outer)
        return out


class TypeEncoding(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim=32):
        """ TypeEncoding layer which takes a 2D-array of measurement type IDs
            of shape (batch_size, timesteps) and convert it into a tensor of
            embeddings of size (batch_size, timesteps, embedding_dim).

            vocab_size:     Number of measurement types to consider
            embedding_dim:  Size of embedding
        """
        super(TypeEncoding, self).__init__()
        self._embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, items):
        """ Converts sequences of IDs into sequences of input embeddings

            items:     Tensor of integer IDs of shape (batch_size, timesteps)

            returns:   Embedding Tensor of shape (batch_size, timesteps, embedding_dim)
        """
        return self._embeddings(items.to(self._device))


if __name__ == '__main__':
    # Sanity Check: does PE look right?
    import matplotlib.pyplot as plt

    t = torch.cumsum(torch.rand(128), dim=0).unsqueeze(0)
    pe = PositionalEncoding(d_model=64, batch_size=1)
    pe_t = pe(t).detach().numpy()

    plt.matshow(pe_t[0])
    plt.title('Positional Encoding')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Timestep')
    plt.show()

    # Sanity check: Does ItemEncoding work?
    x = torch.arange(64 * 18).reshape(64, 18) % 10
    ie = ItemEncoding(vocab_size=10, embedding_dim=16)
    assert ie(x).shape == (64, 18, 16)
