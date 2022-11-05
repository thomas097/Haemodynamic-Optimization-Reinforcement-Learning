import math
import torch
import math
import torch.nn.functional as F


class SinCosPositionalEncoding(torch.nn.Module):
    """ Positional Encoding as used by (Vaswani et al., 2017)
        For details, see https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_dim):
        super(SinCosPositionalEncoding, self).__init__()
        # Precompute 1 / 10000^(2i/d) in log-space
        i = torch.arange(0, embedding_dim, 2).unsqueeze(0).unsqueeze(0)
        div_term = torch.exp(i * (-math.log(10000.0) / embedding_dim))
        self._div_term = torch.nn.Parameter(div_term, requires_grad=False)
        self._embedding_dim = embedding_dim

    def forward(self, t):
        """ Converts a sequence of time steps into a sequence of positional embeddings

            t:       Tensor of real-valued time steps of shape (batch_size, num_steps)

            returns: Embedding Tensor of shape (batch_size, num_steps, embedding_dim)
        """
        pos = t.unsqueeze(2)
        outer_prod = torch.matmul(pos, self._div_term)
        pe = outer_prod.repeat(1, 1, 2)
        pe[:, :, 0::2] = torch.sin(outer_prod)
        pe[:, :, 1::2] = torch.cos(outer_prod)
        return pe


class CategoricalEncoding(torch.nn.Module):
    def __init__(self, embedding_dim, vocab_size):
        """ CategoricalEncoding layer which takes a 2D-array of integer IDs of
            size (batch_size, timesteps) and converts it into a tensor of embeddings
            of shape (batch_size, timesteps, embedding_dim).

            vocab_size:     Size of category vocabulary (i.e. number of items to encode)
            embedding_dim:  Size of output embedding
        """
        super(CategoricalEncoding, self).__init__()
        self._embeddings = torch.nn.Embedding(vocab_size, embedding_dim)

    def forward(self, items):
        """ Converts sequences of IDs into sequences of input embeddings

            items:     Tensor of integer IDs of shape (batch_size, timesteps)

            returns:   Embedding Tensor of shape (batch_size, timesteps, embedding_dim)
        """
        return self._embeddings(items)


if __name__ == '__main__':
    # Sanity check: does PE look right?
    import matplotlib.pyplot as plt

    pe = SinCosPositionalEncoding(embedding_dim=128)

    t = torch.arange(72).float().unsqueeze(0)
    pe_t = pe(t).detach().numpy()

    plt.imshow(pe_t[0])
    plt.ylabel('Timestep')
    plt.xlabel('Embedding Dimension')
    plt.show()

    # Sanity check: do the distances make sense?
    from sklearn.metrics import pairwise_distances

    pe_similarity = 1 - pairwise_distances(pe_t[0], pe_t[0], metric='cosine')

    plt.Figure(figsize=(12, 5))
    plt.title('Cosine similarity between embeddings of positions')
    plt.imshow(pe_similarity)
    plt.show()

    # Sanity check: Does ItemEncoding work?
    x = torch.arange(64 * 18).reshape(64, 18) % 10
    ie = CategoricalEncoding(vocab_size=10, embedding_dim=16)
    assert ie(x).shape == (64, 18, 16)
