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
    def __init__(self, dataset, device, max_len=512, return_history=False):
        dataset = dataset.reset_index(drop=True)
        states = dataset.filter(regex='x\d+').values
        self._states = torch.tensor(states, dtype=torch.float32)

        self._indices = self._get_all_state_indices(dataset)
        self._buffer_size = self._indices.shape[0]
        print('\nEvaluation dataset size: %d states' % self._buffer_size)

        # Build index of states and their histories to speed up replay
        if return_history:
            self._start_of_episode = self._build_history_index(dataset)

        self._return_history = return_history
        self._max_len = max_len
        self._device = device

    @staticmethod
    def _get_all_state_indices(dataset):
        """ Determine indices of visitable states in dataset """
        return dataset.index[dataset.reward.notna()].values

    def _build_history_index(self, dataset):
        """ Builds an index with start/stop indices of histories for each state """
        history_index = {}
        for _, episode in tqdm(dataset.groupby('episode', sort=False), desc='Building index of histories'):
            start_state_index = np.min(episode.index)
            for i in self._get_all_state_indices(episode):
                history_index[i] = start_state_index  # Only need to register first state of episode as end == i
        return history_index

    def _pad_batch_sequences(self, sequences, value=0):
        arr = pad_sequences([s.numpy() for s in sequences], padding="pre", truncating="pre", value=value, dtype=np.float32)
        return torch.tensor(arr)[:, -self._max_len:]

    def iterate(self, batch_size=128):
        for j in range(0, self._buffer_size, batch_size):
            batch_transitions = self._indices[j:j + batch_size]

            states = []
            for i in batch_transitions:
                if self._return_history:
                    states.append(self._states[self._start_of_episode[i]: i + 1])
                else:
                    states.append(self._states[i])

            # If returning histories, consolidate their lengths
            if self._return_history:
                states = self._pad_batch_sequences(states)
            else:
                states = torch.stack(states, dim=0)

            # GPU? GPU!
            yield states.to(self._device)


class PrioritizedReplay:
    """ Implements Prioritized Experience Replay (PER) (Schaul et al., 2016)
        for off-policy RL training from log-data.
        See https://arxiv.org/pdf/1511.05952v3.pdf for details.
    """
    def __init__(self, dataset, device, alpha=0.6, beta0=0.4, eps=1e-2, max_len=512, seed=42, return_history=False):
        # Set random seed
        self._seed = seed
        np.random.seed(seed)

        # Create single DataFrame with episode and timestep information
        self._dataset = dataset.reset_index(drop=True)
        self._dataset.timestep = pd.to_datetime(self._dataset.timestep)

        states = self._dataset.filter(regex='x\d+').values  # state space is marked by 'x*'
        actions = self._dataset.action.values
        rewards = self._dataset.reward.values

        # Move states, action and rewards to GPU if available
        self._states = torch.tensor(states, dtype=torch.float32)
        self._actions = torch.LongTensor(actions).unsqueeze(1)
        self._rewards = torch.LongTensor(rewards).unsqueeze(1)

        # Determine indices of non-terminal states in dataset
        self._indices = self._get_non_terminal_state_indices(self._dataset)

        self._buffer_size = len(self._indices)
        self._TD_errors = np.ones(self._buffer_size) * 1e16  # Big numbers ensure all states are sampled at least once

        # Build index of states and their histories to speed up replay
        if return_history:
            self._history_indices = self._build_history_index(self._dataset)

        self._return_history = return_history
        self._device = device
        self._max_len = max_len
        self._alpha = alpha  # -> 0.0 = to uniform sampling
        self._beta0 = beta0
        self._eps = eps

    @staticmethod
    def _get_non_terminal_state_indices(df):
        """ Determine indices of all visitable non-terminal states in dataset """
        return df.index[df.reward == 0].values

    @staticmethod
    def _build_history_index(df):
        """ Builds an index with start/stop indices of histories for each state """
        history_index = {}
        for _, episode in tqdm(df.groupby('episode', sort=False), desc='Building index of histories'):
            start_index = np.min(episode.index)  # History starts at the beginning of the episode
            state_indices = episode.index[episode.action.notna()]

            for i in range(len(state_indices) - 1):
                state_index = state_indices[i]
                next_state_index = state_indices[i + 1]
                history_index[state_index] = (start_index, next_state_index)  # Define history as start_state:next_state

        return history_index

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

    def sample(self, N):
        # Stochastically sampling of transitions (indices) from replay buffer
        probs = self._selection_probs()
        batch_transitions = np.random.choice(self._indices, size=N, replace=False, p=probs)

        # Compute importance sampling weights for Q-table update
        imp_weights = self._importance_weights(batch_transitions, probs)

        states = []
        actions = []
        rewards = []
        next_states = []

        for i in batch_transitions:
            if self._return_history:
                i_start, i_end = self._history_indices[i]
                states.append(self._states[i_start:i + 1])
                actions.append(self._actions[i])
                rewards.append(self._rewards[i_end])
                next_states.append(self._states[i_start:i_end + 1])
            else:
                states.append(self._states[[i]])
                actions.append(self._actions[i])
                rewards.append(self._rewards[i + 1])
                next_states.append(self._states[[i + 1]])

        # If return_histories, consolidate their lengths
        if self._return_history:
            states = self._pad_batch_sequences(states)
            next_states = self._pad_batch_sequences(next_states)
        else:
            states = torch.stack(states, dim=0)[:, 0]
            next_states = torch.stack(next_states, dim=0)[:, 0]

        # Convert to torch Tensors on right device
        states = states.to(self._device)
        actions = torch.stack(actions, dim=0).to(self._device)
        rewards = torch.stack(rewards, dim=0).to(self._device)
        next_states = next_states.to(self._device)
        imp_weights = torch.Tensor(imp_weights).unsqueeze(1).to(self._device)

        return states, actions, rewards, next_states, batch_transitions, imp_weights
