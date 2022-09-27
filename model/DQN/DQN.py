import copy
import torch

from torch import nn, optim
from tqdm import tqdm
from ExperienceReplay import *


class DuelingDQN(nn.Module):
    """ Implementation of a `dueling` Q-network with separate
        state value and advantage outputs.
    """
    def __init__(self, state_dim=10, num_actions=2, hidden_dims=(16,)):
        super(DuelingDQN, self).__init__()
        shape = (state_dim,) + hidden_dims + (num_actions,)

        # Shared feature network
        layers = []
        for i in range(len(shape) - 2):
            layers.append(nn.Linear(shape[i], shape[i + 1]))
            layers.append(nn.ELU())
        self._base = nn.Sequential(*layers)

        # Value and Advantage nets
        self._value = nn.Linear(shape[-2], 1)
        self._adv = nn.Linear(shape[-2], num_actions)

        # Xavier initialization
        self.apply(self._init_xavier_uniform)

        # Set GPU if available
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

    @staticmethod
    def _init_xavier_uniform(layer):
        # Initialize weights and biases using Xavier initialization
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def sample(self, state):
        # Returns action index with highest π(action|state)
        with torch.no_grad():
            q_vals = self(torch.Tensor(state).unsqueeze(0))
            return torch.argmax(q_vals, dim=1)[0].detach().numpy()

    def forward(self, x):
        h = self._base(x)
        v = self._value(h)
        a = self._adv(h)
        return v + (a - a.mean())


def fit_dueling_double_DQN(model, dataset, state_cols, action_col, reward_col, episode_col, timestep_col, num_episodes=1, alpha=1e-3,
                           gamma=0.99, tau=1e-4, eval_func=None, eval_after=100, batch_size=32, replay_alpha=0.0, replay_beta0=0.4,
                           scheduler_gamma=0.9, step_scheduler_after=100):
    # Load full dataset into buffer
    replay_buffer = PrioritizedExperienceReplay(dataset, episode_col, timestep_col, alpha=replay_alpha, beta0=replay_beta0)

    mse_loss = nn.MSELoss()

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

        avg_mse_loss = 0.0

        # Accumulate gradient over batch
        for i, wis, tr in replay_buffer.sample(N=batch_size):
            # Unpack transition tuple (s, a, r, s')
            state, next_state = torch.Tensor(tr[state_cols].values)
            action, _ = torch.LongTensor(tr[action_col].values)
            reward, _ = torch.Tensor(tr[reward_col].values)

            # Compute expectation and bootstrap Q-target
            q_prev = model(state)[int(action)]

            with torch.no_grad():
                q_next = reward + gamma * model(next_state)[torch.argmax(model(next_state))]

            # Compute loss
            loss = wis * mse_loss(q_prev, q_next) / batch_size
            avg_mse_loss += loss.item()
            loss.backward()

            # TODO: update priority of samples using TD-error

        # Update main model
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
                print('\nEp %s/%s: MSELoss = %.2f, TotalReward = %.2f' % (ep, num_episodes, avg_mse_loss, avg_reward))
            else:
                print('\nEp %s/%s: MSELoss = %.2f' % (ep, num_episodes, avg_mse_loss))
