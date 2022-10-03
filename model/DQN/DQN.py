import copy
import torch

from torch import nn, optim
from tqdm import tqdm
from ExperienceReplay import *


class DuelingLayer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._val = nn.Linear(input_dim, 1)
        self._adv = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        adv = self._adv(x)
        val = self._val(x)
        return val + (adv - torch.mean(adv))


class DuelingDQN(nn.Module):
    """ Implementation of a 'dueling' Deep Q-model (van Hasselt et al., 2015).
        DuelingDQN extends regular DQN by separating the advantage and state
        value streams: Q(s, a) = V(s) + [A(s, a) - mean(A(s, a))].

        For details: https://arxiv.org/pdf/1511.06581.pdf
    """
    def __init__(self, state_dim=10, num_actions=2, hidden_dims=()):
        super(DuelingDQN, self).__init__()

        # Shared feature network
        shape = (state_dim,) + hidden_dims + (num_actions,)
        layers = []
        for i in range(len(shape) - 2):
            layers.append(nn.Linear(shape[i], shape[i + 1]))
            layers.append(nn.LeakyReLU())
        self._base = nn.Sequential(*layers)
        self._head = DuelingLayer(shape[-2], num_actions)

        self.apply(self._init_xavier_uniform)

        # Move to GPU if available
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

    @staticmethod
    def _init_xavier_uniform(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, state):  # -> state.shape = (state_dim,)
        h = self._base(state[None])
        return self._head(h)[0]

    def sample(self, state):
        state = torch.Tensor(state)
        with torch.no_grad():
            action = torch.argmax(self(state))
        return action.detach().numpy()


def fit_dueling_double_DQN(model, dataset, state_cols, action_col, reward_col, episode_col, timestep_col, num_episodes=1, alpha=1e-3,
                           gamma=0.99, tau=1e-2, eval_func=None, eval_after=100, batch_size=32, replay_alpha=0.0, replay_beta0=0.4,
                           scheduler_gamma=0.9, step_scheduler_after=100):
    # Load full dataset into buffer
    replay_buffer = PrioritizedExperienceReplay(dataset, episode_col, timestep_col, alpha=replay_alpha, beta0=replay_beta0)

    huber_loss = nn.HuberLoss()

    # Set Adam optimizer with Stepwise lr schedule
    optimizer = optim.Adam(model.parameters(), lr=alpha)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Copy of model for stabilization
    target = copy.deepcopy(model)
    target.load_state_dict(model.state_dict())

    for ep in tqdm(range(num_episodes)):
        model.train(True)

        #####################
        #     Training      #
        #####################

        avg_loss = 0.0

        for i, wis, tr in replay_buffer.sample(N=batch_size):
            # Unpack transition tuple (s, a, r, s')
            state, next_state = torch.Tensor(tr[state_cols].values)
            action, _ = torch.LongTensor(tr[action_col].values)
            reward, _ = torch.Tensor(tr[reward_col].values)

            # Bootstrap Q-target
            with torch.no_grad():
                next_model_action = torch.argmax(model(next_state))
                q_target = reward + gamma * target(next_state)[next_model_action]

            # Q-estimate of model
            q_prev = model(state)[int(action)]

            # Aggregate loss
            loss = wis * huber_loss(q_prev, q_target) / batch_size
            avg_loss += loss.item()
            loss.backward()

        # Update model
        optimizer.step()
        optimizer.zero_grad()

        # Update target network
        with torch.no_grad():
            for target_w, model_w in zip(target.parameters(), model.parameters()):
                target_w.data = ((1 - tau) * target_w.data + tau * model_w.data).clone()

        # Update lr scheduler
        if ep % step_scheduler_after == 0 and ep > 0:
            scheduler.step()

        ########################
        #      Evaluation      #
        ########################

        if ep % eval_after == 0:

            # If simulator available, validate model over 100 episodes
            if eval_func:
                avg_reward = eval_func(model).groupby(episode_col)[reward_col].sum().mean()
                print('\nEp %s/%s: HuberLoss = %.2f, TotalReward = %.2f' % (ep, num_episodes, avg_loss, avg_reward))
            else:
                print('\nEp %s/%s: HuberLoss = %.2f' % (ep, num_episodes, avg_loss))
