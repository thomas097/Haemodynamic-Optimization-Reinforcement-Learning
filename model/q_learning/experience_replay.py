import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# To ensure backwards-compatibility with older versions of keras.preprocessing
try:
    from keras.preprocessing.sequence import pad_sequences
except:
    from keras_preprocessing.sequence import pad_sequences


class EvaluationReplay:
    """ Implements an experience replay buffer to sequentially iterate over a
        dataset returning batches of individual states or complete histories.
    """
    def __init__(self, dataset, device, max_len=512):
        dataset = dataset.reset_index(drop=True)
        states = dataset.filter(regex='x\d+').values
        self._states = torch.tensor(states, dtype=torch.float32)

        # build index of states and their histories to speed up replay
        self._indices, self._start_of_episode = self._build_history_index(dataset)
        self._indices = self._get_all_state_indices(dataset)
        self._buffer_size = self._indices.shape[0]
        self._max_len = max_len
        self._device = device

    def _build_history_index(self, df):
        """ Builds an index with start/stop indices of histories for each state """
        indices = set()
        history_index = {}
        for _, episode in tqdm(df.groupby('episode', sort=False), desc='Building index of histories'):
            start_index = np.min(episode.index)  # History starts at the beginning of the episode
            state_indices = episode.index[episode.action.notna()]

            for i in range(len(state_indices) - 1):
                state_index = state_indices[i]

                # ensure there's a difference between the state and next state
                # we discard transitions which are very likely to be artifacts of missing data
                diff = torch.sum(torch.absolute(self._states[state_index] - self._states[next_state_index]))
                if diff > 0:
                    history_index[state_index] = start_index
                    indices.add(state_index)

        return np.array(list(indices)), history_index

    def _pad_batch_sequences(self, sequences, value=0):
        arr = pad_sequences([s.numpy() for s in sequences], padding="pre", truncating="pre", value=value, dtype=np.float32)
        return torch.tensor(arr)[:, -self._max_len:]

    def iterate(self, batch_size=128):
        for j in range(0, self._buffer_size, batch_size):
            batch_transitions = self._indices[j:j + batch_size]

            states = []
            for i in batch_transitions:
                states.append(self._states[self._start_of_episode[i]: i + 1])

            # consolidate lengths
            return self._pad_batch_sequences(states).to(self._device)


class PrioritizedReplay:
    def __init__(self, dataset, device, alpha=0.6, beta0=0.4, eps=1e-2, max_len=512, seed=42):
        """ Implements Prioritized Experience Replay (PER) (Schaul et al., 2016)
            for off-policy RL training from log-data.
            See https://arxiv.org/pdf/1511.05952v3.pdf for details.
        """
        self._seed = seed
        np.random.seed(seed)

        # extract states, action and rewards from dataset
        self._dataset = dataset.reset_index(drop=True)
        states = self._dataset.filter(regex='x\d+').values  # state space features are marked by 'x*'
        actions = self._dataset.action.values
        rewards = self._dataset.reward.values

        # convert to PyTorch tensors
        self._states = torch.tensor(states, dtype=torch.float32)
        self._actions = torch.LongTensor(actions).unsqueeze(1)
        self._rewards = torch.LongTensor(rewards).unsqueeze(1)

        # build index of states and their histories to speed up replay
        self._indices, self._history_indices = self._build_history_index(self._dataset)

        # initialize TD-errors as INF so all states are sampled at least once
        self._current = 0
        self._buffer_size = len(self._indices)
        self._TD_errors = np.ones(self._buffer_size) * 1e16

        self._device = device
        self._max_len = max_len
        self._alpha = alpha  # -> 0.0 = to uniform sampling
        self._beta0 = beta0
        self._eps = eps

    def reset(self):
        """ Resets all parameters back to initial values """
        self._current = 0
        self._TD_errors = np.ones(self._buffer_size) * 1e16

    def _build_history_index(self, df):
        """ Builds an index with start/stop indices of histories for each state """
        indices = set()
        history_index = {}
        for _, episode in tqdm(df.groupby('episode', sort=False), desc='Building index of histories'):
            start_index = np.min(episode.index)  # History starts at the beginning of the episode
            state_indices = episode.index[episode.action.notna()]

            for i in range(len(state_indices) - 1):
                state_index = state_indices[i]
                next_state_index = state_indices[i + 1]

                # ensure there's a difference between the state and next state
                # we discard transitions which are very likely to be artifacts of missing data
                diff = torch.sum(torch.absolute(self._states[state_index] - self._states[next_state_index]))
                if diff > 0:
                    history_index[state_index] = (start_index, next_state_index)
                    indices.add(state_index)

        return np.array(list(indices)), history_index

    def _to_index(self, transitions):
        # Returns indices of 'transitions' in full 'self._indices' index
        return [np.where(self._indices == t)[0][0] for t in transitions]

    def _selection_probs(self):
        z = (self._TD_errors + self._eps) ** self._alpha
        return z / np.sum(z)

    def _importance_weights(self, transitions, selection_probs):
        w = (self._buffer_size * selection_probs[self._to_index(transitions)]) ** -self._beta0
        return w / np.max(w)

    def update_priority(self, transitions, td_errors):
        self._TD_errors[self._to_index(transitions)] = np.absolute(td_errors.cpu()).flatten()

    def _pad_batch_sequences(self, sequences, value=0.0):
        arr = pad_sequences([s.numpy() for s in sequences], padding="pre", truncating="pre", value=value, dtype=np.float32)
        return torch.tensor(arr)[:, -self._max_len:]

    def sample(self, N, deterministic=False):
        # Stochastically sampling of transitions (indices) from replay buffer
        probs = self._selection_probs()
        if not deterministic:
            batch_transitions = np.random.choice(self._indices, size=N, replace=False, p=probs)
        else:
            # Iterate over states in episodes in order
            indices = (self._current + np.arange(N)) % self._buffer_size
            self._current = (self._current + N) % self._buffer_size
            batch_transitions = self._indices[indices]

        # Compute importance sampling weights for Q-table update
        imp_weights = self._importance_weights(batch_transitions, probs)

        states = []
        actions = []
        rewards = []
        next_states = []

        for i in batch_transitions:
            i_start, i_end = self._history_indices[i]
            states.append(self._states[i_start:i + 1])
            actions.append(self._actions[i])
            rewards.append(self._rewards[i_end])
            next_states.append(self._states[i_start:i_end + 1])

        # consolidate  lengths
        states = self._pad_batch_sequences(states)
        next_states = self._pad_batch_sequences(next_states)

        # convert to torch Tensors on right device
        states = states.to(self._device)
        actions = torch.stack(actions, dim=0).to(self._device)
        rewards = torch.stack(rewards, dim=0).to(self._device)
        next_states = next_states.to(self._device)
        imp_weights = torch.Tensor(imp_weights).unsqueeze(1).to(self._device)

        return states, actions, rewards, next_states, batch_transitions, imp_weights
