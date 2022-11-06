import os
import copy
import torch
import numpy as np

from tqdm import tqdm
from datetime import datetime
from experience_replay import PrioritizedReplay
from performance_tracking import PerformanceTracker
from loss_functions import *


class DuelingLayer(torch.nn.Module):
    """ Implementation of a `Dueling` layer, separating state-
        value estimation and advantage function as:
        Q(s, a) = V(s) + [A(s, a) - mean_a(A(s, a))]
    """
    def __init__(self, input_dim, output_dim):
        super(DuelingLayer, self).__init__()
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
    def __init__(self, state_dim=10, hidden_dims=(), num_actions=2, disallowed_actions=(), dueling=True):
        super(DQN, self).__init__()
        self.config = locals()

        # Shared feature network
        shape = (state_dim,) + hidden_dims
        layers = []
        for i in range(len(shape) - 1):
            layers.append(torch.nn.Linear(shape[i], shape[i + 1]))
            layers.append(torch.nn.BatchNorm1d(shape[i + 1]))
            layers.append(torch.nn.ELU())
        self._base = torch.nn.Sequential(*layers)

        # Q-head
        params = (shape[-1], num_actions)
        self._head = DuelingLayer(*params) if dueling else torch.nn.Linear(*params)

        # Initialize weights using Xavier initialization
        self.apply(self._init_xavier_uniform)

        # Precompute additive mask to set value of disallowed actions to -inf
        action_mask = torch.zeros((1, num_actions))
        action_mask[0, disallowed_actions] = -1e20
        self._action_mask = torch.nn.Parameter(action_mask, requires_grad=False)

    @staticmethod
    def _init_xavier_uniform(layer):
        """ Init weights using Xavier initialization """
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01)

    def forward(self, states):  # <- (batch_size, state_dim)
        """ Forward pass through DQN """
        h = self._base(states)
        return self._head(h) + self._action_mask

    def q_values(self, states):
        """ Return Q-values of actions given state """
        states = torch.Tensor(states)
        with torch.no_grad():
            qvals = self(states)
        return qvals.cpu().detach().numpy()

    def action_probs(self, states):
        """ Return probability of actions given a state by softmax """
        states = torch.Tensor(states)
        with torch.no_grad():
            actions = torch.softmax(self(states), dim=1)
        return actions.cpu().detach().numpy()

    def sample(self, states):
        """ Sample action given state """
        states = torch.Tensor(states)
        with torch.no_grad():
            actions = torch.argmax(self(states), dim=1)
        return actions.cpu().detach().numpy()


def soft_target_update(target, model, tau):
    """ Performs a soft target network update as p_target = (1- tau) * p_target + tau * p_model
    """
    if target is not None and model is not None:  # Uses encoder?
        with torch.no_grad():
            for target_w, model_w in zip(target.parameters(), model.parameters()):
                target_w.data = ((1 - tau) * target_w.data + tau * model_w.data).clone()


def fit_double_dqn(experiment,
                   policy,
                   dataset,
                   encoder=None,
                   timedelta='4h',
                   num_episodes=1,
                   lrate=1e-3,
                   gamma=0.99,
                   tau=1e-2,
                   replay_alpha=0.4,
                   replay_beta=0.6,
                   batch_size=32,
                   scheduler_gamma=0.9,
                   step_scheduler_after=10000,
                   freeze_encoder=False,
                   lambda_reward=5.0,
                   lambda_phys=0.0,
                   lambda_consv=0.0,
                   eval_func=None,
                   eval_after=1,
                   min_max_reward=(-15, 15),
                   save_on=False):

    # Track performance and hyperparameters of experiment/models
    tracker = PerformanceTracker(experiment)
    tracker.save_experiment_config(policy=policy.config,
                                   encoder=encoder.config if encoder else {'uses_encoder': False},
                                   experiment=locals())

    # GPU? GPU!
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on %s' % device)

    # Load dataset into replay buffer
    replay_buffer = PrioritizedReplay(dataset, alpha=replay_alpha, beta0=replay_beta, timedelta=timedelta,
                                      return_history=encoder is not None, device=device)

    # Adam optimizer with policy and encoder (if provided) and lr scheduler
    modules = torch.nn.ModuleList([policy])
    if encoder is not None:
        modules.append(encoder)
    optimizer = torch.optim.Adam(modules.parameters(), lr=lrate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Use target network for stability
    policy_target = copy.deepcopy(policy)
    encoder_target = copy.deepcopy(encoder)

    # Freeze encoder (optional)
    if encoder is not None and freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        print('Freezing encoder...')

    # Gradient clipping to prevent catastrophic collapse
    for w in policy.parameters():
        if w.requires_grad:
            w.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    #####################
    #     Training      #
    #####################

    # GPU? GPU!
    policy.to(device)
    policy_target.to(device)
    if encoder is not None:
        encoder.to(device)
        encoder_target.to(device)

    # Enable training mode for policy; disable for its target
    policy.train()
    policy_target.eval()
    if encoder is not None:
        encoder.train()
        encoder_target.eval()

    for episode in tqdm(range(num_episodes)):

        # Sample (s, a, r, s') transition from PER
        states, actions, rewards, next_states, transitions, weights = replay_buffer.sample(N=batch_size)

        # Estimate Q(s, a) with s optionally encoded from s_0:t
        states = encoder(states) if encoder is not None else states
        q_vals = policy(states)
        q_pred = q_vals.gather(dim=1, index=actions)

        with torch.no_grad():
            # Bootstrap Q(s', a') with s' optionally a function of s_0:t
            next_states = encoder_target(next_states) if encoder is not None else next_states
            next_target_state_value = policy_target(next_states)

            # Clamp to min/max reward
            next_target_state_value = torch.clamp(next_target_state_value, min=-min_max_reward[0], max=min_max_reward[1])

            # Mask future reward at terminal state
            reward_mask = (rewards == 0).float()

            # Compute target Q value
            next_model_action = torch.argmax(policy(next_states), dim=1, keepdim=True)
            q_target = rewards + gamma * reward_mask * next_target_state_value.gather(dim=1, index=next_model_action)

        # Loss + regularization
        loss = weighted_Huber_loss(q_pred, q_target, weights)
        if lambda_reward > 0:
            loss += lambda_reward * reward_regularizer(q_pred, min_max_reward[1])  # Punishes model for exceeding min/max reward
        if lambda_phys > 0:
            loss += lambbda_phys * physician_regularizer(q_vals, actions)    # Forces decisions to lie close to those of behavior policy
        if lambda_consv > 0:
            loss += lambda_consv * conservative_regularizer(q_vals, q_pred)  # Minimizes Q for OOD actions

        # Q-network and target updates
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Soft target updates
        soft_target_update(policy_target, policy, tau)
        soft_target_update(encoder_target, encoder, tau)

        # Update lrate every few episodes
        if episode % step_scheduler_after == 0 and episode > 0:
            scheduler.step()

        # Update replay priority by TD-error
        td_error = torch.abs(q_target - q_pred).detach()
        replay_buffer.update_priority(transitions, td_error)

        ############################
        #   Performance Tracking   #
        ############################

        tracker.add(loss=loss.item())
        tracker.add(avg_Q_value=torch.mean(q_vals[q_vals > -1e10]).item())  # Drops disallowed actions
        tracker.add(abs_TD_error=torch.mean(td_error).item())
        tracker.add(avg_imp_weight=torch.mean(weights).item())

        # Evaluate model every few episodes
        if episode % eval_after == 0:
            if eval_func:
                eval_args = (encoder, policy) if encoder is not None else (policy,)
                tracker.add(**eval_func(*eval_args))

            tracker.save_metrics()
            print('\nEp %d/%d: %s' % (episode, num_episodes, tracker.print_stats()))

            # Save models upon improvement (or always if save_on=False)
            new_best = tracker.new_best(metric=save_on)
            if new_best:
                print('Model improved! Saving...')

            if not save_on or new_best:
                tracker.save_model_pt(policy, 'policy')
                if encoder:
                    tracker.save_model_pt(encoder, 'encoder')

    # Disable training mode
    policy.eval()
    if encoder:
        encoder.eval()

    print('Done!')
