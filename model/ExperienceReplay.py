import pandas as pd
import numpy as np

from toy_data import discrete_pendulum


class OrderedExperienceReplay:
    """ Implements an ordered Experience Replay (PER) buffer which
        replays experience in the order they occurred.
    """
    def __init__(self, dataset, episode_col='episode', timestep_col='timestep', return_history=False):
        self._df = dataset.reset_index()
        self._episode_col = episode_col
        self._timestep_col = timestep_col

        self._episodes = sorted(list(set(self._df['episode'])))
        self._current_episode = self._episodes[0]

        self._return_history = return_history

    def sample(self, N=None):
        # Determine next episode
        if self._current_episode == self._episodes[-1]:
            self._current_episode = self._episodes[0]
        else:
            i = self._episodes.index(self._current_episode)
            self._current_episode = self._episodes[i + 1]

        # Determine indices from episode
        trans_indices = self._df[self._df[self._episode_col] == self._current_episode]['index'].values[:-1]

        # Optional: Return complete histories
        if self._return_history:
            histories = []
            for i in trans_indices:
                ep = self._df.loc[i][self._episode_col]

                # Extract relevant history of each sampled transition
                episode = self._df[self._df[self._episode_col] == ep]
                history = episode[episode['index'] <= i + 1]
                histories.append((i, 1.0, history))

            return histories

        # Otherwise: Return single transitions
        return [(i, 1.0, self._df.loc[i:i + 1]) for i in trans_indices]


class PrioritizedExperienceReplay:
    """ Implements the Prioritized Experience Replay (PER) buffer
        (Schaul et al., 2016) for off-policy RL training from log-data.
        See: https://arxiv.org/pdf/1511.05952v3.pdf
    """
    def __init__(self, dataset, episode_col='episode', timestep_col='timestep', alpha=0.6, beta0=0.4, return_history=False):
        self._df = dataset.reset_index()
        self._episode_col = episode_col
        self._timestep_col = timestep_col

        # Determines indices of non-terminal states in df
        self._indices = self._non_terminal_indices(self._df)

        self._buffer_size = len(self._indices)
        self._TD_errors = np.ones(self._buffer_size) * 1e16  # Big numbers ensure all are sampled

        self._alpha = alpha  # 0.0 is equiv. to uniform sampling
        self._beta0 = beta0
        self._return_history = return_history

    def _non_terminal_indices(self, df):
        # Extract indices of all non-terminal states
        indices = []
        for _, episode in df.groupby(self._episode_col):
            indices += list(episode['index'].values[:-1])  # [:-1] Ensures terminal states cannot be sampled
        return np.array(indices)

    def _selection_probs(self):
        z = self._TD_errors ** self._alpha
        return z / np.sum(z)

    def _importance_weights(self, chosen_indices, selection_probs):
        idx = [np.where(self._indices == i)[0][0] for i in chosen_indices] # Indices in full index
        w = (self._buffer_size * selection_probs[idx]) ** -self._beta0
        return w / np.max(w)

    def update(self, index, td_error):
        index = np.where(self._indices == index)[0][0]  # Indices in full index
        self._TD_errors[index] = np.absolute(td_error)

    def sample(self, N):
        # Stochastically sampling of transitions (indices) from replay buffer
        probs = self._selection_probs()
        trans_indices = np.random.choice(self._indices, size=N, replace=False, p=probs)

        # Compute importance sampling weights for Q-table update
        imp_weights = self._importance_weights(trans_indices, probs)

        # Optional: Return complete histories
        if self._return_history:
            histories = []
            for i, w in zip(trans_indices, imp_weights):
                ep = self._df.loc[i][self._episode_col]

                # Extract relevant history of each sampled transition
                episode = self._df[self._df[self._episode_col] == ep]
                history = episode[episode['index'] <= i + 1]
                histories.append((i, w, history))

            return histories

        # Otherwise: Return single transitions
        return [(i, w, self._df.loc[i:i + 1]) for i, w in zip(trans_indices, imp_weights)]


if __name__ == '__main__':
    replay_buffer = PriorExpReplay(discrete_pendulum(num_episodes=2), return_history=True)

    for i, w, transition in replay_buffer.sample(N=10):
        print('Index: ', i)
        print('Weight:', w)
        print(transition)
        print()