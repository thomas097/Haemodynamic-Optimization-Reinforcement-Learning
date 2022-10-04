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
        return val + (adv - torch.mean(adv, dim=1, keepdim=True))


class DuelingDQN(nn.Module):
    """ Implementation of a 'dueling' Deep Q-model (van Hasselt et al., 2015).
        DuelingDQN extends regular DQN by separating advantage and state value
        streams as Q(s, a) = V(s) + [A(s, a) - mean_a(A(s, a))].

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

        # Initialize weights using Xavier initialization
        self.apply(self._init_xavier_uniform)

        # Move to GPU if available
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

    @staticmethod
    def _init_xavier_uniform(layer):
        if isinstance(layer, nn.Linear):
            nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, states):  # -> state.shape = (batch_size, state_dim,)
        h = self._base(states)
        return self._head(h)

    def sample(self, states):
        states = torch.Tensor(states)
        with torch.no_grad():
            actions = torch.argmax(self(states))
        return actions.detach().numpy()


# Computed HuberLoss with additional (importance) weight assigned to samples
def weighted_Huber_loss(x, y, weights, delta=1.0):
    a = 0.5 * torch.pow(x - y, 2)
    b = delta * (torch.abs(x - y) - 0.5 * delta)
    mask = (torch.absolute(x - y) < delta).float()
    hubert_loss = mask * a + (1 - mask) * b
    return torch.mean(weights * hubert_loss)


def fit_dueling_double_DQN(model, dataset, state_cols, action_col, reward_col, episode_col, timestep_col, num_episodes=1,
                           alpha=1e-3, gamma=0.99, tau=1e-2, eval_func=None, eval_after=100, batch_size=32,
                           replay_alpha=0.0, replay_beta0=0.4, scheduler_gamma=0.9, step_scheduler_after=100):
    # Load full dataset into buffer
    replay_buffer = PrioritizedExperienceReplay(dataset, state_cols, action_col, reward_col, episode_col,
                                                timestep_col, alpha=replay_alpha, beta0=replay_beta0)

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

        # Sample batch from experience replay
        states, actions, rewards, next_states, trans_indices, imp_weights = replay_buffer.sample(N=batch_size)

        # Bootstrap Q-targets
        with torch.no_grad():
            next_model_action = torch.argmax(model(next_states), dim=1, keepdim=True)
            q_target = rewards + gamma * target(next_states).gather(dim=1, index=next_model_action)

        # Q-estimates of model for states
        q_prev = model(states).gather(dim=1, index=actions)

        # Estimate loss
        loss = weighted_Huber_loss(q_prev, q_target, imp_weights)

        # Aggregate loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO: update importance weights

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
                print('\nEp %s/%s: HuberLoss = %.2f, TotalReward = %.2f' % (ep, num_episodes, loss.item(), avg_reward))
            else:
                print('\nEp %s/%s: HuberLoss = %.2f' % (ep, num_episodes, loss.item()))
