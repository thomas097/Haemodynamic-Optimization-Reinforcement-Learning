import copy
import torch
from torch import nn, optim
from tqdm import tqdm
from ExperienceReplay import PrioritizedExperienceReplay, OrderedExperienceReplay


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

    def sample(self, state):
        # Returns action index with highest π(action|state)
        q_vals = self(torch.Tensor(state)[None])
        return torch.argmax(q_vals, dim=1)[0]

    def copy_weights(self, other):
        for param, param_value in zip(self.parameters(), other.parameters()):
            param.data.copy_(param_value.data)
        return self

    def forward(self, x):
        h = self._base(x)
        v = self._V(h)
        a = self._A(h)
        return v + (a - a.mean())


def fit_dueling_double_DQN(model, dataset, state_cols, action_col, reward_col, episode_col, timestep_col, alpha=1e-4,
                           gamma=0.9, tau=1e-2, dataset_passes=100, replay_size=96, eval_func=None, replay_alpha=0.6, replay_beta0=0.4):
    # Load full dataset into buffer
    # replay_buffer = PrioritizedExperienceReplay(dataset, episode_col, timestep_col, alpha=replay_alpha, beta0=replay_beta0)
    replay_buffer = OrderedExperienceReplay(dataset, episode_col, timestep_col)

    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=alpha)

    target = copy.deepcopy(model).copy_weights(model)

    num_episodes = len(set(dataset[episode_col]))
    print('\nTraining on %s episodes for %s iterations...\n' % (num_episodes, dataset_passes))

    target.train(False)
    model.train(True)

    for it in range(dataset_passes):

        ######################
        #     Training
        ######################

        total_loss = 0
        total_steps = 0

        for _ in tqdm(range(num_episodes)):
            optimizer.zero_grad()

            for i, wis, tr in replay_buffer.sample(N=replay_size):
                # Unpack transition tuple (s, a, r, s')
                state, next_state = torch.Tensor(tr[state_cols].values)
                action = torch.LongTensor(tr[action_col].values)[0]
                reward = torch.Tensor(tr[reward_col].values)[0]

                # Initial estimate of Q(s, a)
                q_prev = model(state)[action]

                # Updated Q(s, a)_new = R + gamma * imp_weight * Q(s', argmax_a[Q'(s', a)])
                q_new = reward + gamma * target(next_state)[torch.argmax(model(next_state))]

                # Compute squared TD-error (accounting for importance weight)
                loss = mse_loss(q_prev, q_new) * wis
                total_loss += loss.item()
                total_steps += 1
                loss.backward()

                # Update replay priority
                # td_error = abs(float(q_prev - q_new))
                # replay_buffer.update(i, td_error)

            # Update model weights
            optimizer.step()

            # Update target weights
            for target_params, model_params in zip(target.parameters(), model.parameters()):
                target_params.data.copy_(model_params.data * tau + target_params.data * (1.0 - tau))

        ###############################
        #     Evaluation (optional)
        ###############################

        if eval_func:
            avg_reward = eval_func(model).groupby(episode_col)[reward_col].sum().mean()
            print('\nIteration %s: Avg. Reward = %s\n' % (it, avg_reward))
        else:
            print('\nIteration %s: Avg. Sqr. TD-error = %s\n' % (it, total_loss / total_steps))
