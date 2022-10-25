import pandas as pd
import numpy as np
import torch

# Backward-compatibility with previous versions of keras
try:
    from keras.preprocessing.sequence import pad_sequences
except:
    from keras_preprocessing.sequence import pad_sequence


class EvaluationReplay:
    """ Implements a simple experience replay buffer which sequentially
        iterates over a dataset returning batches of histories.
    """
    def __init__(self, dataset, max_len=512, return_history=False):
        # Move states to GPU if available
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        dataset = dataset.reset_index(drop=True)
        states = dataset.filter(regex='x\d+').values
        self._states = torch.Tensor(states).to(self._device)
        self._episodes = dataset.episode.values
        self._timesteps = dataset.timestep.values

        self._indices = self._non_terminal_indices(dataset)
        self._buffer_size = self._indices.shape[0]
        print('Evaluation dataset size: %d states' % self._buffer_size)

        self._return_history = return_history
        self._max_len = max_len

    @staticmethod
    def _non_terminal_indices(df):
        # NaNs in 'reward' column mark absorbing terminal states
        indices = []
        for _, episode in df.groupby('episode'):
            indices += episode.index[episode.reward.notna()].to_list()
        return np.array(indices)

    def _consolidate_length(self, sequences, value=0):
        arr = pad_sequences(sequences, dtype=np.float32, padding="pre", truncating="pre", value=value)
        return torch.Tensor(arr)[:, -self._max_len:]

    def iterate(self, batch_size=128):
        for j in range(0, self._buffer_size, batch_size):
            batch_transitions = self._indices[j:j + batch_size]

            states = []
            for i in batch_transitions:
                # Extract states in the same episode `ep` up to and including timestep `ts`
                if self._return_history:
                    ep = self._episodes[i]
                    ts = self._timesteps[i]
                    i_history = (self._episodes == ep) & (self._timesteps <= ts)
                else:
                    i_history = i
                states.append(self._states[i_history])

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
    def __init__(self, dataset, alpha=0.6, beta0=0.4, eps=1e-2, dt='4h', max_len=512, return_history=False):
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

        # USed to find states prior to t and dt ahead (next step)
        self._timestep = pd.to_datetime(self._dataset.timestep).values
        self._episodes = self._dataset.episode.values

        self._buffer_size = len(self._indices)
        self._TD_errors = np.ones(self._buffer_size) * 1e16  # Big numbers ensure all states are sampled at least once

        self._alpha = alpha  # alpha = 0.0 -> to uniform sampling
        self._beta0 = beta0
        self._eps = eps
        self._return_history = return_history
        self._max_len = max_len
        self._dt = pd.to_timedelta(dt)

    @staticmethod
    def _non_terminal_indices(df):
        # NaNs in 'reward' column mark absorbing terminal states
        indices = []
        for _, episode in df.groupby('episode'):
            indices += episode.index[episode.reward.notna()].to_list()
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

    def _consolidate_length(self, sequences, value=0):
        arr = pad_sequences(sequences, dtype=np.float32, padding="pre", truncating="pre", value=value)
        return torch.Tensor(arr)[:, -self._max_len:]

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
            # Extract states preceding transition `i` at time step `ts` in episode `ep` and next state at `ts + dt`
            if self._return_history:
                ep = self._episodes[i]
                ts = self._timestep[i]
                i_history = (self._episodes == ep) & (self._timestep <= ts + self._dt)
            else:
                i_history = np.array([i, i + 1])  # -> history + next state!

            states.append(self._states[i_history][:-1])
            actions.append(self._actions[i])
            rewards.append(self._rewards[i])
            next_states.append(self._states[i_history][1:])

        # If return_histories, consolidate their lengths
        if self._return_history:
            states = self._consolidate_length(states)
            next_states = self._consolidate_length(next_states)
        else:
            states = torch.stack(states, dim=0)[:, 0]  # If no history, no need to keep track of temporal dimension
            next_states = torch.stack(next_states, dim=0)[:, 0]

        # Convert to torch Tensors and enable GPU devices
        actions = torch.stack(actions, dim=0)
        rewards = torch.stack(rewards, dim=0)
        imp_weights = torch.Tensor(imp_weights).unsqueeze(1).to(self._device)

        return states, actions, rewards, next_states, batch_transitions, imp_weights
