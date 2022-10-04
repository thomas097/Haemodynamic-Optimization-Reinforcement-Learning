import pandas as pd
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


# class OrderedExperienceReplay:
#     """ Implements an ordered Experience Replay (PER) buffer which
#         replays experience in the order they occurred.
#     """
#     def __init__(self, dataset, episode_col='episode', timestep_col='timestep', return_history=False):
#         self._df = dataset.reset_index()
#         self._episode_col = episode_col
#         self._timestep_col = timestep_col
#
#         self._episodes = sorted(list(set(self._df[episode_col])))
#         self._current_episode = self._episodes[0]
#         print('Running %s episodes' % len(self._episodes))
#
#         self._return_history = return_history
#
#     def sample(self, N=None):
#         # Determine next episode
#         if self._current_episode == self._episodes[-1]:
#             self._current_episode = self._episodes[0]
#             print('End-of-buffer: Starting over!')
#         else:
#             i = self._episodes.index(self._current_episode)
#             self._current_episode = self._episodes[i + 1]
#
#         # Determine indices from episode
#         trans_indices = self._df[self._df[self._episode_col] == self._current_episode]['index'].values[:-1]
#
#         # Optional: Return complete histories
#         if self._return_history:
#             histories = []
#             for i in trans_indices:
#                 ep = self._df.loc[i][self._episode_col]
#
#                 # Extract relevant history of each sampled transition
#                 episode = self._df[self._df[self._episode_col] == ep]
#                 history = episode[episode['index'] <= i + 1]
#                 histories.append((i, 1.0, history))
#
#             return histories
#
#         # Otherwise: Return single transitions
#         return [(i, 1.0, self._df.loc[i:i + 1]) for i in trans_indices]


class PrioritizedExperienceReplay:
    """ Implements the Prioritized Experience Replay (PER) buffer
        (Schaul et al., 2016) for off-policy RL training from log-data.
        See: https://arxiv.org/pdf/1511.05952v3.pdf
    """
    def __init__(self, dataset, state_cols, action_col='action', reward_col='reward', episode_col='episode',
                 timestep_col='timestep', alpha=0.6, beta0=0.4, return_history=False):
        self._df = dataset.reset_index()
        self._state_cols = state_cols
        self._action_col = action_col
        self._reward_col = reward_col
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
            indices += list(episode['index'].values[:-1])  # [:-1] ensures terminal states are not sampled
        return np.array(indices)

    def _selection_probs(self):
        z = self._TD_errors ** self._alpha
        return z / np.sum(z)

    def _importance_weights(self, chosen_indices, selection_probs):
        idx = [np.where(self._indices == i)[0][0] for i in chosen_indices]  # Indices in full index
        w = (self._buffer_size * selection_probs[idx]) ** -self._beta0
        return w / np.max(w)

    def update(self, indices, td_errors):
        idx = [np.where(self._indices == i)[0][0] for i in indices]  # Indices in full index
        self._TD_errors[idx] = np.absolute(td_errors)

    @staticmethod
    def _consolidate_length(histories):
        histories = [torch.Tensor(seq) for seq in histories]
        return pad_sequence(histories, batch_first=True, padding_value=0.0)

    def sample(self, N):
        # Stochastically sampling of transitions (indices) from replay buffer
        probs = self._selection_probs()
        trans_indices = np.random.choice(self._indices, size=N, replace=False, p=probs)

        # Compute importance sampling weights for Q-table update
        imp_weights = self._importance_weights(trans_indices, probs)

        states = []
        actions = []
        rewards = []
        next_states = []

        for i in trans_indices:
            # What episode does transition `i` belong to?
            ep = self._df.loc[i][self._episode_col]

            if self._return_history:
                # Extract relevant history of transition `i` and next state
                episode = self._df[self._df[self._episode_col] == ep]
                history = episode[episode['index'] <= i + 1][self._state_cols].values
            else:
                # Extract just current and next state
                history = self._df.loc[i: i + 1][self._state_cols].values

            state = history[:-1]
            action = self._df.loc[i][self._action_col]
            reward = self._df.loc[i][self._reward_col]
            next_state = history[1:]

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)

        # If return_histories, consolidate their lengths
        if self._return_history:
            states = self._consolidate_length(states)
            next_states = self._consolidate_length(next_states)
        else:
            states = torch.Tensor(np.array(states))[:, 0]  # If no history, no need to keep track of temporal dimension
            next_states = torch.Tensor(np.array(next_states))[:, 0]

        # convert to torch Tensors
        actions = torch.LongTensor(actions).unsqueeze(1)  # TODO: allow gpu devices!
        rewards = torch.Tensor(rewards).unsqueeze(1)
        imp_weights = torch.Tensor(imp_weights)

        return states, actions, rewards, next_states, trans_indices, imp_weights


if __name__ == '__main__':
    dataset = pd.DataFrame({'episode': np.repeat(np.arange(10), 10),
                            'timestep': np.tile(np.arange(10), 10),
                            'state': np.random.random(100),
                            'action': np.random.random(100),
                            'reward': np.random.random(100)})
    replay_buffer = PrioritizedExperienceReplay(dataset, return_history=True)

    for i, w, transition in replay_buffer.sample(N=10):
        print('Index: ', i)
        print('Weight:', w)
        print(transition)
        print()