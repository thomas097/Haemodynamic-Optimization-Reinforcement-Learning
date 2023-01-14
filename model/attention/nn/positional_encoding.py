import math
import numpy as np
import torch


class PositionalEncoding(torch.nn.Module):
    """ Positional Encoding as used by (Vaswani et al., 2017)
        For details, please refer to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, embedding_dim):
        super(PositionalEncoding, self).__init__()
        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

        self._embedding_dim = embedding_dim
        i = torch.arange(0, embedding_dim, 2, device=self._device).unsqueeze(0).unsqueeze(0)
        self._div_term = torch.exp(i * (-math.log(10000.0) / embedding_dim))

    def forward(self, t):
        """ Converts a sequence of time steps into a sequence of positional embeddings

            t:       Tensor of real-valued time steps of shape (batch_size, num_steps)

            returns: Embedding Tensor of shape (batch_size, num_steps, embedding_dim)
        """
        position = t.to(self._device).unsqueeze(2)
        div_term = self._div_term.repeat(position.shape[0], 1, 1)  # Account for varying batch sizes!

        pe = torch.zeros(*t.shape, self._embedding_dim, device=self._device)
        outer_prod = torch.bmm(position, div_term)
        pe[:, :, 0::2] = torch.sin(outer_prod)
        pe[:, :, 1::2] = torch.cos(outer_prod)
        return pe



if __name__ == '__main__':
    # Sanity Check: does PE look right?
    import matplotlib.pyplot as plt

    t = torch.arange(128).float().unsqueeze(0)
    pe = PositionalEncoding(embedding_dim=64)
    pe_t = pe(t.float()).detach().numpy()[0]

    plt.matshow(pe_t)
    plt.title('Positional Encoding')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Timestep')
    plt.show()

    # Cosine distance?
    pe_t = pe_t / np.linalg.norm(pe_t, axis=1, keepdims=True)
    dist_mat = pe_t.dot(pe_t.T)
    plt.matshow(dist_mat)
    plt.title('Dot product - Positional Encoding')
    plt.xlabel('Keys')
    plt.ylabel('Queries')
    plt.show()



    
    
