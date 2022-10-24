import pandas as pd
import numpy as np
import torch
from torch.nn.functional import pad


class EvaluationReplay:
    """ Implements a simple experience replay buffer which sequentially
        iterates over a dataset returning batches of states.
    """
    def __init__(self, dataset, return_history=False):
        # Extract states and episodes from dataset
        states = dataset.filter(regex='x\d+').values
        episodes = dataset.episode.values

        # Move states to GPU if available
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._states = torch.Tensor(states).to(self._device)

        self._episodes = episodes
        self._indices = np.arange(episodes.shape[0])
        self._buffer_size = episodes.shape[0]
        self._return_history = return_history

    @staticmethod
    def _consolidate_length(histories, val=0.0):  # TODO: sloooow but no built-in for this!
        max_len = max([len(h) for h in histories])
        padded_histories = [pad(h, (0, 0, max_len - len(h), 0), value=val).unsqueeze(0) for h in histories]
        return torch.cat(padded_histories, dim=0)

    def iterate(self, batch_size=128):
        for j in range(0, self._buffer_size, batch_size):
            # Batch of transitions
            transitions = self._indices[j:j + batch_size]

            states = []
            for i in transitions:
                # Extract states up to and including transition `i` in the same episode
                if self._return_history:
                    states.append(self._states[(self._episodes == self._episodes[i]) & (self._indices <= i)])
                else:
                    states.append(self._states[i])  # ith state only

            # If return_histories, consolidate their lengths
            if self._return_history:
                yield self._consolidate_length(states)
            else:
                yield torch.stack(states, dim=0)  # If no history, no need to keep track of temporal dim 2


class PrioritizedReplay:
    """ Implements Prioritized Experience Replay (PER) (Schaul et al., 2016)
        for off-policy RL training from log-data.
        See https://arxiv.org/pdf/1511.05952v3.pdf for details.
    """
    def __init__(self, dataset, alpha=0.6, beta0=0.4, eps=1e-2, return_history=False):
        # Create single DataFrame with episode and timestep information.
        self._dataset = dataset.reset_index(drop=True)
        states = self._dataset.filter(regex='x\d+').values  # state space is marked by 'x*'
        actions = self._dataset.action.values
        rewards = self._dataset.reward.values

        # Move states, action and rewards to GPU if available
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._states = torch.Tensor(states).to(self._device)
        self._actions = torch.LongTensor(actions).unsqueeze(1).to(self._device)
        self._rewards = torch.LongTensor(rewards).unsqueeze(1).to(self._device)

        # Determines indices of non-terminal states in df
        self._indices = self._non_terminal_indices(self._dataset)

        self._buffer_size = len(self._indices)
        self._TD_errors = np.ones(self._buffer_size) * 1e16  # Big numbers ensure all states are sampled at least once

        self._alpha = alpha  # alpha = 0.0 -> to uniform sampling
        self._beta0 = beta0
        self._eps = eps
        self._return_history = return_history

    @staticmethod
    def _non_terminal_indices(df):
        # Extract indices of all non-terminal states
        indices = []
        for _, episode in df.groupby('episode'):
            nan_rewards = episode['reward'].isna()  # NaN rewards indicate absorbing terminal states!
            indices += episode.index[~nan_rewards].to_list()
        return np.array(indices)

    def _to_index(self, transitions):
        # Returns indices of `idx` in full `self._indices` index
        return [np.where(self._indices == t)[0][0] for t in transitions]

    def _selection_probs(self):
        z = (self._TD_errors + self._eps) ** self._alpha
        return z / np.sum(z)

    def _importance_weights(self, transitions, selection_probs):
        w = (self._buffer_size * selection_probs[self._to_index(transitions)]) ** -self._beta0
        return w / np.max(w)

    def update_priority(self, transitions, td_errors):
        self._TD_errors[self._to_index(transitions)] = np.absolute(td_errors.cpu()).flatten()

    @staticmethod
    def _consolidate_length(histories, val=0.0):  # TODO: sloooow but no built-in for this!
        max_len = max([len(h) for h in histories])
        padded_histories = [pad(h, (0, 0, max_len - len(h), 0), value=val).unsqueeze(0) for h in histories]
        return torch.cat(padded_histories, dim=0)

    def sample(self, N):
        # Stochastically sampling of transitions (indices) from replay buffer
        probs = self._selection_probs()
        transitions = np.random.choice(self._indices, size=N, replace=False, p=probs)

        # Compute importance sampling weights for Q-table update
        imp_weights = self._importance_weights(transitions, probs)

        states = []
        actions = []
        rewards = []
        next_states = []

        for i in transitions:

            # Extract states preceding transition `i` in episode + next state
            if self._return_history:
                ep = self._dataset.loc[i]['episode']
                history_idx = ((self._dataset['episode'] == ep) & (self._dataset.index <= i + 1)).values
            else:
                history_idx = np.array([i, i + 1])  # -> history + next state!

            states.append(self._states[history_idx][:-1])
            actions.append(self._actions[i])
            rewards.append(self._rewards[i])
            next_states.append(self._states[history_idx][1:])

        # If return_histories, consolidate their lengths
        if self._return_history:
            states = self._consolidate_length(states)
            next_states = self._consolidate_length(next_states)
        else:
            states = torch.stack(states, dim=0)[:, 0]  # If no history, no need to keep track of temporal dim 2
            next_states = torch.stack(next_states, dim=0)[:, 0]

        # Convert to torch Tensors and enable GPU devices
        actions = torch.stack(actions, dim=0)
        rewards = torch.stack(rewards, dim=0)
        imp_weights = torch.Tensor(imp_weights).unsqueeze(1).to(self._device)

        return states, actions, rewards, next_states, transitions, imp_weights
