import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class PrioritizedReplay:
    """ Implements Prioritized Experience Replay (PER) (Schaul et al., 2016)
        for off-policy RL training from log-data.
        See https://arxiv.org/pdf/1511.05952v3.pdf for details.
    """
    def __init__(self, states, actions, rewards, episodes, timesteps, alpha=0.6, beta0=0.4, return_history=False):
        # Create single DataFrame with episode and timestep information.
        self._df = pd.concat([episodes, timesteps], axis=1).reset_index(drop=True)
        self._df.columns = ['episode', 'timestep']

        # Move states, action and rewards to GPU if available
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self._states = torch.Tensor(states.values).to(self._device)
        self._actions = torch.LongTensor(actions.values).unsqueeze(1).to(self._device)
        self._rewards = torch.LongTensor(rewards.values).unsqueeze(1).to(self._device)

        # Determines indices of non-terminal states in df
        self._indices = self._non_terminal_indices(self._df)

        self._buffer_size = len(self._indices)
        self._TD_errors = np.ones(self._buffer_size) * 1e16  # Big numbers ensure all states are sampled at least once

        self._alpha = alpha  # alpha = 0.0 -> to uniform sampling
        self._beta0 = beta0
        self._return_history = return_history

    @staticmethod
    def _non_terminal_indices(df):
        # Extract indices of all non-terminal states
        indices = []
        for _, episode in df.groupby('episode'):
            indices += list(episode.index.values[:-1])  # [:-1] ensures terminal states are not sampled
        return np.array(indices)

    def _to_index(self, transitions):
        # Returns indices of `idx` in full `self._indices` index
        return [np.where(self._indices == t)[0][0] for t in transitions]

    def _selection_probs(self):
        z = self._TD_errors ** self._alpha
        return z / np.sum(z)

    def _importance_weights(self, transitions, selection_probs):
        w = (self._buffer_size * selection_probs[self._to_index(transitions)]) ** -self._beta0
        return w / np.max(w)

    def update_priority(self, transitions, td_errors):
        self._TD_errors[self._to_index(transitions)] = np.absolute(td_errors.cpu()).flatten()

    @staticmethod
    def _consolidate_length(histories):
        return pad_sequence(histories, batch_first=True, padding_value=0.0)

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
                ep = self._df.loc[i]['episode']
                history_idx = ((self._df['episode'] == ep) & (self._df.index <= i + 1)).values
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
