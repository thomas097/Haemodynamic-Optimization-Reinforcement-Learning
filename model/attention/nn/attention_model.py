import torch
import numpy as np

from time import time_ns
from attention_layers import AttentionBlock


class PositionalEncoding(torch.nn.Module):
    """ Implements absolute positional encoding (Vaswani et al., 2017)
    """
    def __init__(self, embedding_dim=32, max_len=256):
        super().__init__()
        self._max_length = max_len
        self._embedding_dim = embedding_dim

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, t):
        """ Converts a sequence of time steps into a sequence of positional embeddings

            t:       Ndarray of real-valued time steps of shape (batch_size, num_steps)

            returns: Embedding Tensor of shape (batch_size, num_steps, embedding_dim)
        """
        # Create wk matrix of size embedding_dim / 2
        i = torch.arange(0, self._embedding_dim // 2)
        wk = 1 / (10000 ** (2 * i / self._embedding_dim)).unsqueeze(0).unsqueeze(0)

        # Extend t with embedding dimension and compute sin/cos embeddings
        pe = torch.Tensor(t).unsqueeze(2).repeat(1, 1, self._embedding_dim)
        pe[:, :, 0::2] = torch.sin(pe[:, :, 0::2] * wk)
        pe[:, :, 1::2] = torch.cos(pe[:, :, 1::2] * wk)
        return pe.to(self._device)


class ItemEncoding(torch.nn.Module):
    def __init__(self, vocab, embedding_dim=32):
        """ Implementation of an Item Encoding layer which takes a 2D-array
            of categorical items and convert it into fixed embeddings.

            vocab:          Vocabulary of items to consider
            embedding_dim:  Number of embedding dimensions
        """
        super().__init__()
        # Mapping from items to unique IDs
        self._item_ids = {item: i for i, item in enumerate(vocab)}
        self._item_ids['[CLS]'] = len(vocab)

        # Defines the item embedding layer
        self._item_embeddings = torch.nn.Embedding(len(vocab) + 1, embedding_dim)
        self._embedding_dim = embedding_dim

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, items):
        """ Converts a sequence of items into a sequence of input embeddings

            items:     Ndarray of integer item_ids of shape (batch_size, num_items)

            returns:   Embedding Tensor of shape (batch_size, num_items, embedding_dim)
        """
        # Convert object items to their unique ids
        item_ids = torch.LongTensor([[self._item_ids[it] for it in seq] for seq in items]).to(self._device)
        return self._item_embeddings(item_ids)


class Transformer(torch.nn.Module):
    """ Implementation of the Transformer (Vaswani et al., 2017)
    """
    def __init__(self, item_vocab, d_model, n_blocks=1, k=4, mask_layers=(0,)):
        super().__init__()
        # Input Transformation and Attention Blocks
        self._pos_encoding = PositionalEncoding(embedding_dim=d_model)
        self._item_encoding = ItemEncoding(vocab=item_vocab, embedding_dim=d_model)
        self._blocks = torch.nn.ModuleList([AttentionBlock(d_model=d_model, k=k) for _ in range(n_blocks)])

        # Layers to mask
        self._mask_layers = mask_layers

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, items, time_steps, add_cls=True):
        # Add optional [CLS] token to sequences
        if add_cls:
            items = np.array([list(seq) + ['[CLS]'] for seq in items])
            time_steps = np.array([list(seq) + [np.max(seq)] for seq in time_steps])  # Assume max time step is decision time

        # Define mask
        items_rep = np.repeat(items[:, None], repeats=items.shape[1], axis=1)
        mask = torch.BoolTensor(np.transpose(items_rep, (0, 2, 1)) == items_rep)

        # Compute sum of item embedding and positional embedding
        item_embedding = self._item_encoding(items)
        pos_embedding = self._pos_encoding(time_steps)
        embedding = item_embedding + pos_embedding

        for i, block in enumerate(self._blocks):
            embedding = block(embedding, mask=mask if i in self._mask_layers else None)

        return embedding


if __name__ == '__main__':
    # Example dataset
    vocab = ['a', 'b', 'c', 'd', 'e']
    items = np.random.choice(vocab, size=(32, 1000))
    times = np.cumsum(np.random.random(items.shape), axis=1)
    print('In: items.shape = %s  times.shape = %s' % (items.shape, times.shape))

    # Model
    model = Transformer(item_vocab=vocab, d_model=32)

    # Timing test
    time1 = time_ns()
    out = model(items=items, time_steps=times)
    time2 = time_ns()
    print('Out:', out.shape)

    print('Time elapsed:', (time2 - time1) / 1e9)