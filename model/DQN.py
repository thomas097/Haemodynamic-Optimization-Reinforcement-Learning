import copy
import torch
from torch import nn, optim
from tqdm import tqdm

from ExperienceReplay import PriorExpReplay, OrderedExpReplay
from toy_data import cartpole


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
            layers.append(nn.Sigmoid())
        self.__base = nn.Sequential(*layers)

        self.__V = nn.Linear(shape[-2], 1)
        self.__A = nn.Linear(shape[-2], shape[-1])

    def copy_weights(self, other):
        for param, param_value in zip(self.parameters(), other.parameters()):
            param.data.copy_(param_value.data)
        return self

    def forward(self, x):
        h = self.__base(x)
        v = self.__V(h)
        a = self.__A(h)
        return v + (a - a.mean())


def fit_DuelingDQN(actor,
                   dataset,
                   state_cols=('state_0', 'state_1'),
                   action_col='action',
                   reward_col='reward',
                   episode_col='episode',
                   timestep_col='timestep',
                   alpha=1e-2,
                   gamma=0.9,
                   tau=1e-2,
                   passes=2,
                   replay_size=18,
                   replay_alpha=0.6,
                   replay_beta0=0.4):

        print('Loading DataFrame into replay buffer')
        # replay_buffer = PriorExpReplay(dataset=dataset,
        #                                episode_col=episode_col,
        #                                timestep_col=timestep_col,
        #                                alpha=replay_alpha,
        #                                beta0=replay_beta0,
        #                                return_history=False)

        replay_buffer = OrderedExpReplay(dataset, episode_col, timestep_col)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(actor.parameters(), lr=alpha)

        # Create target network
        target = copy.deepcopy(actor).copy_weights(actor)

        num_episodes = len(set(dataset[episode_col]))
        print('Training on %s episodes %s times...' % (num_episodes, passes))

        for pass_ in range(passes):
            total_loss = 0

            for ep in range(num_episodes):
                for i, w, tr in replay_buffer.sample(N=replay_size):
                    # Unpack transition
                    state, next_state = torch.Tensor(tr[state_cols].values)
                    action = torch.LongTensor(tr[action_col].values)[0]
                    reward = torch.Tensor(tr[reward_col].values)[0]

                    # Initial estimate of Q(s, a)
                    q_prev = actor(state)

                    # Q(s, a)_{new} = R + gamma * Q(s_{t+1}, argmax_a(Q'(s_t+1, a)))
                    q_new = q_prev.clone()
                    q_new[action] = reward + gamma * target(next_state)[torch.argmax(actor(next_state))]

                    # Update actor
                    loss = criterion(q_prev, q_new)
                    total_loss += loss.item()

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # Update target
                    for target_p, actor_p in zip(target.parameters(), actor.parameters()):
                        target_p.data.copy_(actor_p.data * tau + target_p.data * (1.0 - tau))

            print('Pass %s: MSE = %s' % (pass_, total_loss))


if __name__ == '__main__':
    STATE_COLS = ['state_0', 'state_1']
    ACTION_COL = 'action'
    REWARD_COL = 'reward'
    HIDDEN_DIMS = (32, 32,)

    actor = DuelingDQN(state_dim=len(STATE_COLS),
                       num_actions=2,
                       hidden_dims=HIDDEN_DIMS)

    dataset = cartpole(num_episodes=1000)

    fit_DuelingDQN(actor=actor, dataset=dataset, state_cols=STATE_COLS, action_col=ACTION_COL,
                   reward_col=REWARD_COL, episode_col='episode', timestep_col='timestep')


