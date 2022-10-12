import os
import copy
import torch
import numpy as np

from tqdm import tqdm
from experience_replay import PrioritizedReplay
from performance_tracking import PerformanceTracker
from loss_functions import weighted_Huber_loss, reward_regularizer


class DuelingLayer(torch.nn.Module):
    """ Implementation of a `Dueling` layer, separating state-
        value estimation and advantage function as:
        Q(s, a) = V(s) + [A(s, a) - mean_a(A(s, a))]
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self._val = torch.nn.Linear(input_dim, 1)
        self._adv = torch.nn.Linear(input_dim, output_dim)

    def forward(self, x):
        adv = self._adv(x)
        val = self._val(x)
        return val + (adv - torch.mean(adv, dim=1, keepdim=True))


class DQN(torch.nn.Module):
    """ Implementation of a DQN model (Mnih et al., 2013). By default, DQN will
        employ a dueling architecture as proposed in (van Hasselt et al., 2015).
        For details: https://arxiv.org/pdf/1511.06581.pdf
    """
    def __init__(self, state_dim=10, hidden_dims=(), num_actions=2, dueling=True):
        super(DQN, self).__init__()

        # Shared feature network
        shape = (state_dim,) + hidden_dims
        layers = []
        for i in range(len(shape) - 1):
            layers.append(torch.nn.Linear(shape[i], shape[i + 1]))
            layers.append(torch.nn.LeakyReLU())
        self._base = torch.nn.Sequential(*layers)

        if dueling:
            self._head = DuelingLayer(shape[-1], num_actions)
        else:
            self._head = torch.nn.Linear(shape[-1], num_actions)

        # Initialize weights using Xavier initialization
        self.apply(self._init_xavier_uniform)

        # Move to GPU if available
        self._device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.to(self._device)

    @staticmethod
    def _init_xavier_uniform(layer):
        """ Init weights using Xavier initialization """
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, states):  # -> state.shape = (batch_size, state_dim,)
        """ Forward pass through DQN """
        h = self._base(states)
        return self._head(h)

    def action_probs(self, states):
        """ Return probability of action given state as given by softmax """
        states = torch.Tensor(states)
        with torch.no_grad():
            actions = torch.softmax(self(states), dim=1)
        return actions.detach().numpy()

    def sample(self, states):
        """ Sample action given state """
        states = torch.Tensor(states)
        with torch.no_grad():
            actions = torch.argmax(self(states), dim=1)
        return actions.detach().numpy()


def fit_double_dqn(experiment, policy, states, actions, rewards, episodes, timesteps, num_episodes=1, alpha=1e-3,
                   gamma=0.99, tau=1e-2, lamda=1e-3, eval_func=None, eval_after=100, batch_size=32, replay_params=(0.0, 0.4),
                   scheduler_gamma=0.9, step_scheduler_after=100, encoder=None, freeze_encoder=False, reward_clipping=np.inf):

    # Upload dataset to experience buffer
    replay_buffer = PrioritizedReplay(states, actions, rewards, episodes, timesteps, alpha=replay_params[0],
                                      beta0=replay_params[1], return_history=bool(encoder))

    optimizer = torch.optim.Adam(policy.parameters(), lr=alpha)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_gamma)
    tracker = PerformanceTracker(experiment)

    # Use target network for stability
    target = copy.deepcopy(policy)
    target.load_state_dict(policy.state_dict())  # Sanity check

    # Freeze encoder (optional)
    if encoder and freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        print('Freezing encoder...')

    # Gradient clipping to prevent catastrophic collapse
    for w in policy.parameters():
        w.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    #####################
    #     Training      #
    #####################

    # Enable training mode
    policy.train()
    if encoder:
        encoder.train()

    for episode in tqdm(range(num_episodes)):

        # Sample (s, a, r, s') transition from PER
        states, actions, rewards, next_states, transitions, weights = replay_buffer.sample(N=batch_size)

        # Encode states if encoder provided
        if encoder:
            states = encoder(states)
            next_states = encoder(next_states)

        # Compute Q-estimate
        q_pred = policy(states).gather(dim=1, index=actions)

        # Bootstrap Q-target
        with torch.no_grad():
            next_model_action = torch.argmax(policy(next_states), dim=1, keepdim=True)
            q_target = rewards + gamma * target(next_states).gather(dim=1, index=next_model_action)

        # Loss + regularization
        loss = weighted_Huber_loss(q_pred, q_target, weights)
        loss += lamda * reward_regularizer(q_pred, reward_clipping)

        # Policy update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Soft target update
        with torch.no_grad():
            for target_w, policy_w in zip(target.parameters(), policy.parameters()):
                target_w.data = ((1 - tau) * target_w.data + tau * policy_w.data).clone()

        # Update lrate every few episodes
        if episode % step_scheduler_after == 0 and episode > 0:
            scheduler.step()

        # Update replay priority by TD-error
        td_error = torch.abs(q_target - q_pred).detach()
        replay_buffer.update_priority(transitions, td_error)

        ############################
        #   Performance Tracking   #
        ############################

        tracker.add('loss', loss.item())
        tracker.add('avg_Q', torch.mean(policy(states)).item())
        tracker.add('abs_TD_error', torch.mean(td_error))

        if episode % eval_after == 0:
            if eval_func:
                eval_args = (policy,) if encoder is None else (encoder, policy)
                for metric, value in eval_func(*eval_args).items():
                    tracker.add(metric, value)

            print('\nEp %s/%s: %s' % (episode, num_episodes, tracker.print_stats()))

        tracker.save_metrics()

    # Disable training mode
    policy.eval()
    if encoder:
        encoder.eval()

    # Save policy (and encoder) to `model` directory
    tracker.save_model_pt(policy, 'policy')
    if encoder:
        tracker.save_model_pt(encoder, 'encoder')
