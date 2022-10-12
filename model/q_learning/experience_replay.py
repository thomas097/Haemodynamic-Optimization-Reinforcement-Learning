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
        # Create single DataFrame out of states, actions, rewards, etc.
        self._df = pd.concat([states, actions, rewards, episodes, timesteps], axis=1)

        # Assign canonical column names
        self._state_cols = ['s%s' % i for i in range(states.shape[1])] # 's41' -> 42nd state-space feature
        self._df.columns = self._state_cols + ['action', 'reward', 'episode', 'timestep']

        # Determines indices of non-terminal states in df
        self._indices = self._non_terminal_indices(self._df)

        self._buffer_size = len(self._indices)
        self._TD_errors = np.ones(self._buffer_size) * 1e16  # Big numbers ensure all are sampled

        self._alpha = alpha  # 0.0 is equiv. to uniform sampling
        self._beta0 = beta0
        self._return_history = return_history

    @staticmethod
    def _non_terminal_indices(df):
        # Extract indices of all non-terminal states
        indices = []
        for _, episode in df.groupby('episode'):
            indices += list(episode.index.values[:-1])  # [:-1] ensures terminal states are not sampled
        return np.array(indices)

    def _selection_probs(self):
        z = self._TD_errors ** self._alpha
        return z / np.sum(z)

    def _importance_weights(self, chosen_indices, selection_probs):
        idx = [np.where(self._indices == i)[0][0] for i in chosen_indices]  # Indices in full index
        w = (self._buffer_size * selection_probs[idx]) ** -self._beta0
        return w / np.max(w)

    def update_priority(self, indices, td_errors):
        idx = [np.where(self._indices == i)[0][0] for i in indices]  # Equiv. to self._indices.index(i)
        self._TD_errors[idx] = np.absolute(td_errors).flatten()

    @staticmethod
    def _consolidate_length(histories):
        histories = [torch.Tensor(seq) for seq in histories]
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
                history = self._df[(self._df['episode'] == ep) & (self._df.index <= i + 1)][self._state_cols].values
            else:
                history = self._df.loc[i: i + 1][self._state_cols].values  # -> history + next state!

            states.append(history[:-1])
            actions.append(int(self._df.loc[i]['action']))
            rewards.append(self._df.loc[i]['reward'])
            next_states.append(history[1:])

        # If return_histories, consolidate their lengths
        if self._return_history:
            states = self._consolidate_length(states)
            next_states = self._consolidate_length(next_states)
        else:
            states = torch.Tensor(np.array(states))[:, 0]  # If no history, no need to keep track of temporal dimension
            next_states = torch.Tensor(np.array(next_states))[:, 0]

        # convert to torch Tensors
        actions = torch.LongTensor(actions).unsqueeze(1)  # TODO: enable gpu devices!
        rewards = torch.Tensor(rewards).unsqueeze(1)
        imp_weights = torch.Tensor(imp_weights).unsqueeze(1)

        return states, actions, rewards, next_states, transitions, imp_weights


if __name__ == '__main__':
    # Create fake dataset of 10 episodes, each of 10 timesteps. The agent
    # observes 32-dim states and can take 25 actions
    episodes = np.repeat(np.arange(10), 10)
    timesteps = np.tile(np.arange(10), 10)
    states = np.random.random((100, 32))
    actions = np.random.randint(0, 25, 100)
    rewards = np.random.random(100)

    # Load into replay buffer
    replay_buffer = PrioritizedReplay(states, actions, rewards, episodes, timesteps, return_history=True)

    states, actions, rewards, next_states, transitions, weights = replay_buffer.sample(N=10)
    print('states.shape:     ', states.shape)
    print('actions.shape:    ', actions.shape)
    print('rewards.shape:    ', rewards.shape)
    print('next_states.shape:', next_states.shape)
    print('transitions.shape:', transitions.shape)
    print('weights.shape:    ', weights.shape)