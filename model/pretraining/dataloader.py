import torch
import random
import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self, dataset, device, seed=42, padding=0.0, nan=-1):
        # Store data under 'x*' columns as ndarray
        dataset = dataset.reset_index(drop=True)
        self._states = dataset.filter(regex='x\d+').values
        self._actions = np.nan_to_num(dataset.action.values, nan=nan)[:, np.newaxis]  # |A| x 1

        # Create index of episodes in dataset
        self._indices = self._index_episodes(dataset)
        self._index_size = len(self._indices)

        self._nan = nan
        self._padding = padding
        self._device = device
        random.seed(seed)

    @staticmethod
    def _index_episodes(df):
        """ Creates index of tuples marking beginning and end of each episode """
        index = []
        for _, episode_df in df.groupby('episode', sort=False):
            state_indices = episode_df.index
            index.append((state_indices[0], state_indices[-1]))  # first to last state of episode
        return index

    def _pad_sequences(self, sequences, value=None):
        """ Consolidates length of variable-length sequences using constant padding """
        value = self._padding if value is None else value
        maxlen = max([s.shape[0] for s in sequences])
        return np.array([np.pad(s, ((maxlen - s.shape[0], 0), (0, 0)), constant_values=value) for s in sequences])

    def iterate(self, batch_size, shuffle=True):
        """ Iterates over batches of data in dataset """
        assert batch_size > 0

        if shuffle:
            random.shuffle(self._indices)

        for b in range(0, self._index_size, batch_size):

            episodes = []
            actions = []
            for start, end in self._indices[b: b + batch_size]:
                episodes.append(self._states[start: end + 1])
                actions.append(self._actions[start: end + 1])

            # Pad with zeros to consolidate their lengths
            episodes = self._pad_sequences(episodes)
            actions = self._pad_sequences(actions, value=self._nan)[:, :, 0]

            # Return Tensor on right device
            states = torch.Tensor(episodes).to(self._device)
            actions = torch.LongTensor(actions).to(self._device)
            yield states, actions


if __name__ == '__main__':
    train_loader = DataLoader(pd.read_csv('../../preprocessing/datasets/mimic-iii/attention_4h/mimic-iii_train.csv'))
    for x, a in train_loader.iterate(batch_size=64):
        print(x.shape, a.shape)
        break
