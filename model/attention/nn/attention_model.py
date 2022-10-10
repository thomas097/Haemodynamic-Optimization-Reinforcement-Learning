import torch
import numpy as np
from attention import MultiHeadAttention


class PositionalItemEncoding(torch.nn.Module):
    """ Implementation of a Positional Item Encoding (PIE) layer which takes
        series of (item, timestep) pairs and converts it into a fixed embeddings.
    """
    def __init__(self, items, embedding_dim=32):
        super().__init__()
        # Mapping from items to unique IDs
        self._item_ids = {item: i for i, item in enumerate(items)}

        # Defines the item embedding layer
        self._item_embeddings = torch.nn.Embedding(len(items), embedding_dim)
        self._embedding_dim = embedding_dim

    def forward(self, items, timesteps):
        """ Converts a sequence of items and real-valued timesteps
            into a sequence of input embeddings

            items:     Ndarray of integer item_ids of shape (batch_size, num_items)
            timesteps: Ndarray of real-valued time steps of shape (batch_size, num_items)

            returns:   Embedding Tensor of shape (batch_size, num_items, embedding_dim)
        """
        # Convert object items to unique ids
        item_ids = torch.LongTensor([[self._item_ids[it] for it in seq] for seq in items])

        # Convert item_ids to embeddings
        item_embeddings = self._item_embeddings(item_ids)

        # Compute positional encoding of t



if __name__ == '__main__':
    # Example dataset
    items = np.array([['a', 'b', 'c', 'a', 'd'],
                      ['b', 'b', 'a', 'a', 'e'],
                      ['c', 'c', 'e', 'b', 'a']])
    times = np.cumsum(np.random.random(items.shape), axis=1)

    # Encode sequence
    pie = PositionalItemEncoding(items=['a', 'b', 'c', 'd', 'e'])
    pie(items=items, timesteps=times)