import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

# make backward-compatible with older versions of keras_preprocessing
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
        self._indices, self._histories = self._build_history_index(dataset)
        self._buffer_size = self._indices.shape[0]
        self._max_len = max_len
        self._device = device

    def _build_history_index(self, df):
        """ Builds an index with start/stop indices of histories for each state """
        indices = set()
        histories = dict()
        for _, episode in tqdm(df.groupby('episode', sort=False), desc='Building index'):
            # history starts at the beginning of the episode
            start_index = np.min(episode.index)
            state_indices = episode.index[episode.action.notna()]

            for state_index in state_indices:
                histories[state_index] = start_index
                indices.add(state_index)

        # create index
        indices = np.array(list(indices))

        return indices, histories

    def _pad_batch_sequences(self, sequences, value=0):
        arr = pad_sequences([s.numpy() for s in sequences], padding="pre", truncating="pre", value=value, dtype=np.float32)
        return torch.tensor(arr)[:, -self._max_len:]

    def iterate(self, batch_size=128):
        with torch.no_grad():
            for j in range(0, self._buffer_size, batch_size):
                batch_transitions = self._indices[j:j + batch_size]

                states = []
                for i in batch_transitions:
                    states.append(self._states[self._histories[i]: i + 1])

                # consolidate lengths
                states = self._pad_batch_sequences(states).to(self._device)
                yield states


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
        self._dataset['reward'] = self._dataset.groupby('episode').reward.transform(self._smear_reward)
        states = self._dataset.filter(regex='x\d+').values  # state space features are marked by 'x*'
        actions = self._dataset.action.values
        rewards = self._dataset.reward.values

        # convert to PyTorch tensors
        self._states = torch.tensor(states).float()
        self._actions = torch.tensor(actions).long().unsqueeze(1)
        self._rewards = torch.tensor(rewards).float().unsqueeze(1)

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

    def _smear_reward(self, r, scale=1):
        r_terminal = r.values[-1]
        decay = np.exp(scale * (np.arange(len(r)) - len(r) + 1))
        return (decay * r_terminal).astype(np.float32)

    def reset(self):
        """ Resets all parameters back to initial values """
        self._current = 0
        self._TD_errors = np.ones(self._buffer_size) * 1e16

    def _build_history_index(self, df):
        """ Builds an index with start/stop indices of histories for each state """
        indices = set()
        history_index = {}
        for _, episode in tqdm(df.groupby('episode', sort=False), desc='Building index'):
            start_index = np.min(episode.index)  # History starts at the beginning of the episode
            state_indices = episode.index[episode.action.notna()]

            for i in range(len(state_indices) - 1):
                state_index = state_indices[i]
                next_state_index = state_indices[i + 1]

                history_index[state_index] = (start_index, next_state_index)
                indices.add(state_index)

        indices = np.array(list(indices))
        return indices, history_index

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
        with torch.no_grad():
            # Stochastically sampling of transitions (indices) from replay buffer
            probs = self._selection_probs()
            if deterministic:
                # Iterate over states in episodes in order
                idx = (self._current + np.arange(N)) % self._buffer_size
                self._current = (self._current + N) % self._buffer_size
                batch_transitions = self._indices[idx]
            else:
                # Sample transitions acc. to priority
                batch_transitions = np.random.choice(self._indices, size=N, replace=False, p=probs)

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
            imp_weights = torch.tensor(imp_weights).float().unsqueeze(1).to(self._device)

            return states, actions, rewards, next_states, batch_transitions, imp_weights
