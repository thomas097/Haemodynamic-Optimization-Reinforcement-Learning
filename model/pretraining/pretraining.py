import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm


class DataLoader:
    def __init__(self, data_file, device='cpu', seed=42):
        # Store data under x* columns as ndarray
        df = pd.read_csv(data_file).reset_index(drop=True)
        self._data = df.filter(regex='x\d+').values

        # Create index of episodes in dataset
        self._indices = self._index_episodes(df)
        self._index_size = len(self._indices)
        self._device = device

        random.seed(seed)

    @staticmethod
    def _index_episodes(df):
        """ Creates index of tuples marking beginning and end of each episode """
        index = []
        with tqdm(desc="Building index...") as pbar:
            for _, episode_df in df.groupby('episode', sort=False):
                state_indices = episode_df.index
                index.append((state_indices[0], state_indices[-1]))  # first to last state of episode
                pbar.update(1)
        return index

    @staticmethod
    def _pad_sequences(sequences, value=0.0):
        """ Consolidates length of variable-length sequences using constant padding """
        maxlen = max([s.shape[0] for s in sequences])
        return np.array([np.pad(s, ((maxlen - s.shape[0], 0), (0, 0)), constant_values=value) for s in sequences])

    def iterate(self, batch_size, shuffle=True):
        """ Iterates over batches of data in dataset """
        if shuffle:
            random.shuffle(self._indices)

        for b in range(0, self._index_size, batch_size):

            episodes = []
            for start, end in self._indices[b: b + batch_size]:
                episodes.append(self._data[start: end + 1])

            # Pad with zeros to consolidate their lengths
            episodes = self._pad_sequences(episodes)

            # Return Tensor on right device
            yield torch.Tensor(episodes).to(self._device)


if __name__ == '__main__':
    train_loader = DataLoader('../../preprocessing/datasets/mimic-iii/attention_4h/mimic-iii_train.csv')
    for x in train_loader.iterate(batch_size=64):
        for y in x[:, :, 0]:
            print(y)
        break
