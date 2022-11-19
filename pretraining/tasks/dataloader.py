import torch
import random
import numpy as np
import pandas as pd
from tqdm import tqdm

# To make backwards-compatible older versions of keras.preprocessing
try:
    from keras.preprocessing.sequence import pad_sequences
except:
    from keras_preprocessing.sequence import pad_sequences


class DataLoader:
    """ Implements a simple data loader returning sequences of states (histories)
        and their corresponding actions or labels
    """
    def __init__(self, dataset, device, label_col='action', maxlen=512):
        # Store dataset of states and actions
        dataset = dataset.reset_index(drop=True)
        self._states = dataset.filter(regex='x\d+').values
        self._labels = dataset[label_col]

        self._indices = self._get_all_state_indices(dataset)
        self._buffer_size = self._indices.shape[0]
        print('\nDataset size: %d states' % self._buffer_size)

        # Build index of state:history pairs to speed up history lookup
        self._start_of_episode = self._build_history_index(dataset)

        self._maxlen = maxlen
        self._device = device

    @staticmethod
    def _get_all_state_indices(dataset):
        """ Determine indices of visitable states in dataset """
        return dataset.index[dataset.action.notna()].values

    def _build_history_index(self, dataset):
        """ Builds an index with start/stop indices of histories for each state """
        history_index = {}
        for _, episode in tqdm(dataset.groupby('episode', sort=False), desc='Building index of histories'):
            start_state_index = np.min(episode.index)
            for i in self._get_all_state_indices(episode):
                history_index[i] = start_state_index  # Only need to register first state of episode as end == i
        return history_index

    def _pad_batch_sequences(self, sequences, value=0):
        """ Pads sequences in list [s0, s1, ..., sn] to equal length """
        arr = pad_sequences(sequences, padding="pre", value=value, dtype=np.float32)
        return torch.tensor(arr)[:, -self._maxlen:]

    def iterate(self, batch_size, shuffle=False):
        if shuffle:
            random.shuffle(self._indices)

        for j in range(0, self._buffer_size, batch_size):
            batch_transitions = self._indices[j:j + batch_size]

            states, labels = [], []
            for i in batch_transitions:
                states.append(self._states[self._start_of_episode[i]: i + 1])
                labels.append(self._labels[i])

            states = self._pad_batch_sequences(states).to(self._device)
            labels = torch.tensor(labels).long().to(self._device)
            yield states, labels
