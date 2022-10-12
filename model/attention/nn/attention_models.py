import torch
import numpy as np

from time import time_ns
from attention_layers import SelfAttentionBlock
from positional_encoding import ItemEncoding, PositionalEncoding


class SelfAttentionModel(torch.nn.Module):
    """ Implementation of a self-attention model (Vaswani et al., 2017)
    """
    def __init__(self, in_features, d_model, n_blocks=1, k=4):
        super().__init__()
        self._pos_encoding = PositionalEncoding(d_model)
        self._input_encoding = torch.nn.Linear(in_features, d_model)

        encoder_blocks = [SelfAttentionBlock(d_model, k=k) for _ in range(n_blocks)]
        self._model = torch.nn.Sequential(*encoder_blocks)

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, x, return_last=True):
        # Positional embeddings for positions in X
        indices = np.arange(x.shape[1])[np.newaxis]
        pos_embedding = torch.flip(self._pos_encoding(indices), dims=[1])  # reverse timesteps to make T-1 the anchor

        # Run model on input with positional embedding
        inputs = self._input_encoding(x) + pos_embedding
        y = self._model(inputs)

        # Optionally return only last step
        return y[:, -1] if return_last else y


class ItemWiseTransformer(torch.nn.Module):
    """ Implementation of the Transformer (Vaswani et al., 2017) using
        an ItemEncoder to circumvent the need for binning.
    """
    def __init__(self, item_vocab, d_model, n_blocks=1, k=4, mask_layers=(0,)):
        super().__init__()
        self._pos_encoding = PositionalEncoding(d_model)
        self._item_encoding = ItemEncoding(item_vocab, d_model)
        self._blocks = torch.nn.ModuleList([SelfAttentionBlock(d_model, k=k) for _ in range(n_blocks)])

        # Layers to mask
        self._mask_layers = mask_layers

        # Use GPU if available
        self._device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self._device)

    def forward(self, items, time_steps, add_cls=True, return_last=True):
        # Add optional [CLS] token to sequences
        if add_cls:
            items = np.array([list(seq) + ['[CLS]'] for seq in items])
            time_steps = np.array([list(seq) + [torch.max(seq)] for seq in time_steps])  # Assume max time step is decision time

        # Define mask
        items_rep = np.repeat(items[:, None], repeats=items.shape[1], axis=1)
        mask = torch.BoolTensor(np.transpose(items_rep, (0, 2, 1)) == items_rep)

        # Compute sum of item embedding and positional embedding
        item_embedding = self._item_encoding(items)
        pos_embedding = torch.flip(self._pos_encoding(time_steps), dims=[1])  # reverse timesteps to make T-1 the anchor
        inputs = item_embedding + pos_embedding

        for i, block in enumerate(self._blocks):
            inputs = block(inputs, mask=mask if i in self._mask_layers else None)

        # Optionally return only last step
        return inputs[:, -1] if return_last else inputs


if __name__ == '__main__':
    # Datasets
    vocab = ['a', 'b', 'c', 'd', 'e']
    item_dataset = (np.random.choice(vocab, size=(32, 1000)),
                    torch.cumsum(torch.rand((32, 1000)), dim=1))
    regular_dataset = (torch.randn(32, 100, 5),)

    # Models
    regular = SelfAttentionModel(in_features=5, d_model=32)
    itemwise = ItemWiseTransformer(item_vocab=vocab, d_model=32)

    # Sanity check: timing test
    def timing_test(model, inputs, N=1):
        time1 = time_ns()
        for i in range(N):
            out = model(*inputs)
        time2 = time_ns()
        return (time2 - time1) / (1e9 * N)


    print('(Regular)  Time elapsed: %.3f' % timing_test(regular, inputs=regular_dataset))
    print('(Itemwise) Time elapsed: %.3f' % timing_test(itemwise, inputs=item_dataset))