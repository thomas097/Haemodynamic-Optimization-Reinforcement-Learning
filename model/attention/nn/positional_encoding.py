import torch


class PositionalEncoding(torch.nn.Module):
    """ Implementation of the Positional Encoding used by (Vaswani et al., 2017)
        For details, please refer to https://arxiv.org/pdf/1706.03762.pdf
    """
    def __init__(self, d_model=32, max_len=256):
        super().__init__()
        self._max_length = max_len
        self._d_model = d_model

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, t):
        """ Converts a sequence of time steps into a sequence of positional embeddings

            t:       Ndarray of real-valued time steps of shape (batch_size, num_steps)

            returns: Embedding Tensor of shape (batch_size, num_steps, embedding_dim)
        """
        # Create wk matrix of size (1, 1, embedding_dim / 2)
        i = torch.arange(0, self._d_model // 2)
        wk = 1 / (10000 ** (2 * i / self._d_model)).unsqueeze(0).unsqueeze(0)

        # Extend t with embedding dimension and compute sin/cos embeddings
        pe = torch.Tensor(t).unsqueeze(2).repeat(1, 1, self._d_model)
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
        self._item_embeddings = torch.nn.Embedding(len(self._item_ids), embedding_dim)
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