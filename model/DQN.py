import copy
import torch
from torch import nn, optim
from tqdm import tqdm

from ExperienceReplay import PrioritizedExperienceReplay, OrderedExperienceReplay
from toy_data import discrete_pendulum


class DuelingDQN(nn.Module):
    """ Implementation of a `dueling` Q-network with separate
        state value and advantage outputs.
    """
    def __init__(self, state_dim=10, num_actions=2, hidden_dims=(16,)):
        super(DuelingDQN, self).__init__()

        shape = (state_dim,) + hidden_dims + (num_actions,)

        layers = []
        for i in range(len(shape) - 2):
            layers.append(nn.Linear(shape[i], shape[i + 1]))
            layers.append(nn.LeakyReLU())
        self._base = nn.Sequential(*layers)

        self._V = nn.Linear(shape[-2], 1)
        self._A = nn.Linear(shape[-2], shape[-1])

    def copy_weights(self, other):
        for param, param_value in zip(self.parameters(), other.parameters()):
            param.data.copy_(param_value.data)
        return self

    def forward(self, x):
        h = self._base(x)
        v = self._V(h)
        a = self._A(h)
        return v + (a - a.mean())


def fit_dueling_double_DQN(model, dataset, state_cols, action_col, reward_col, episode_col, timestep_col,
                           alpha=1e-4, gamma=0.9, tau=1e-2, passes=10, replay_size=200, replay_alpha=0.6, replay_beta0=0.4):

    # Load dataset into experience replay buffer
    replay_buffer = PrioritizedExperienceReplay(dataset, episode_col, timestep_col, alpha=replay_alpha, beta0=replay_beta0)

    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=alpha)

    # Set up target network
    target = copy.deepcopy(model).copy_weights(model)

    num_episodes = len(set(dataset[episode_col]))
    print('Training on %s episodes %s times...' % (num_episodes, passes))

    target.train(False)
    model.train(True)

    for pass_ in range(passes):
        total_loss = 0
        total_steps = 0

        for _ in tqdm(range(num_episodes)):

            for i, w, tr in replay_buffer.sample(N=replay_size):
                # Unpack transition tuple (s, a, r, s')
                state, next_state = torch.Tensor(tr[state_cols].values)
                action = torch.LongTensor(tr[action_col].values)[0]
                reward = torch.Tensor(tr[reward_col].values)[0]

                # Initial estimate of Q(s, a)
                q_prev = model(state)[action]

                # Updated Q(s, a)_new = R + gamma * imp_weight * Q(s', argmax_a[Q'(s', a)])
                q_new = reward + gamma * target(next_state)[torch.argmax(model(next_state))]

                # Compute squared TD-error
                loss = mse_loss(q_prev, q_new)
                total_loss += loss.item()
                total_steps += 1

                td_error = abs(float(q_prev - q_new))
                replay_buffer.update(i, td_error)

                # Update model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Update target (parameterized by theta1)
                for target_params, model_params in zip(target.parameters(), model.parameters()):
                    target_params.data.copy_(model_params.data * tau + target_params.data * (1.0 - tau))

        print('Pass %s: Avg. Sqr. TD-error = %s\n' % (pass_, total_loss / total_steps))


if __name__ == '__main__':
    STATE_COLS = ['state_0', 'state_1', 'state_2']
    HIDDEN_DIMS = (16, 16,)

    ACTION_COL = 'action'
    NUM_ACTIONS = 5

    REWARD_COL = 'reward'
    EPISODE_COL = 'episode'
    TIMESTEP_COL = 'timestep'

    model = DuelingDQN(state_dim=len(STATE_COLS),
                       num_actions=NUM_ACTIONS,
                       hidden_dims=HIDDEN_DIMS)

    dataset = discrete_pendulum(num_episodes=50)

    fit_dueling_double_DQN(model, dataset, state_cols=STATE_COLS, action_col=ACTION_COL, reward_col=REWARD_COL,
                           episode_col=EPISODE_COL, timestep_col=TIMESTEP_COL)


