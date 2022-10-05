"""
Author:   Thomas Bellucci
Filename: DQN_roggeveen_et_al.py
Descr.:   Performs the training of a Dueling Double DQN model with state-space
          encoder over entire histories.
Date:     01-10-2022
"""

import torch
import copy
import pandas as pd

from tqdm import tqdm
from DQN import DuelingDQN, weighted_Huber_loss
from ckconv_v3 import CKConv
from ExperienceReplay import PrioritizedExperienceReplay
from PerformanceTracking import PerformanceTracker


# Temporary encoder model
class CKCNN(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._conv1 = CKConv(in_channels, out_channels)
        self._conv2 = CKConv(out_channels, out_channels)
        self._activation = torch.nn.ELU()

    def forward(self, x):
        h = self._activation(self._conv1(x))
        y = self._activation(self._conv2(h))
        return y[:, -1]  # Return representation of final states


# Modified `fit_dueling_double_dqn()` returning whole histories
def fit_dueling_double_DQN_with_history(model, encoder, dataset, state_cols, action_col, reward_col, episode_col,
                                        timestep_col, num_episodes=1, alpha=1e-3, gamma=0.99, tau=1e-2, eval_func=None,
                                        eval_after=100, batch_size=32, replay_alpha=0.0, replay_beta0=0.4,
                                        scheduler_gamma=0.9, step_scheduler_after=100, freeze_encoder=False):
    # Load full dataset into buffer
    replay_buffer = PrioritizedExperienceReplay(dataset, state_cols, action_col, reward_col, episode_col, timestep_col,
                                                alpha=replay_alpha, beta0=replay_beta0, return_history=True)

    tracker = PerformanceTracker()

    # Set Adam optimizer with Stepwise lr schedule
    optimizer = torch.optim.Adam(model.parameters(), lr=alpha)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=scheduler_gamma)

    # Copy of model for stabilization
    target = copy.deepcopy(model)
    target.load_state_dict(model.state_dict())

    # freeze encoder parameters (optional)
    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    # Gradient clipping
    for w in model.parameters():
        w.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    for ep in tqdm(range(num_episodes)):
        model.train(True)

        #####################
        #     Training      #
        #####################

        # Sample batch from experience replay
        histories, actions, rewards, next_histories, trans_indices, imp_weights = replay_buffer.sample(N=batch_size)

        # Feed histories through encoder
        states = encoder(histories)
        next_states = encoder(next_histories)

        # Bootstrap Q-targets
        with torch.no_grad():
            next_model_action = torch.argmax(model(next_states), dim=1, keepdim=True)
            q_target = rewards + gamma * target(next_states).gather(dim=1, index=next_model_action)

        # Q-estimates of model for states
        model_q_values = model(states)
        q_prev = model_q_values.gather(dim=1, index=actions)

        # Estimate loss
        loss = weighted_Huber_loss(q_prev, q_target, imp_weights)

        # Aggregate loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update target network
        with torch.no_grad():
            for target_w, model_w in zip(target.parameters(), model.parameters()):
                target_w.data = ((1 - tau) * target_w.data + tau * model_w.data).clone()

        # Update lr scheduler every few episodes
        if ep % step_scheduler_after == 0 and ep > 0:
            scheduler.step()

        # TD-error
        td_error = torch.abs(q_target - q_prev)
        # TODO: update importance weights

        #######################
        #       Metrics       #
        #######################

        tracker.add('huber_loss', loss.item())
        tracker.add('avg_Q_value', torch.mean(model_q_values).item())
        tracker.add('avg_best_Q_value', torch.mean(torch.max(model_q_values, dim=1)[0]).item())
        tracker.add('abs_TD_error', torch.mean(td_error))

        if ep % eval_after == 0:

            # If simulator available, validate model over 100 episodes
            if eval_func:
                avg_reward = eval_func(model).groupby(episode_col)[reward_col].sum().mean()
                tracker.add('avg_reward', avg_reward)

            print('\nEp %s/%s: %s' % (ep, num_episodes, tracker.print_stats()))

        tracker.save()


if __name__ == '__main__':
    # Define columns marking state- and action-space
    STATE_SPACE_FEATURES = ['max_vp', 'total_iv_fluid', 'sirs_score', 'sofa_score', 'weight', 'ventilator', 'height',
                            'age', 'gender', 'heart_rate', 'temp', 'mean_bp', 'dias_bp', 'sys_bp', 'resp_rate', 'spo2',
                            'natrium', 'chloride', 'kalium', 'trombo', 'leu', 'anion_gap', 'aptt', 'art_ph', 'asat',
                            'alat', 'bicarbonaat', 'art_be', 'ion_ca', 'lactate', 'paco2', 'pao2', 'hb', 'bilirubin',
                            'creatinine', 'inr', 'ureum', 'albumin', 'magnesium', 'calcium', 'glucose',
                            'total_urine_output']
    ACTION_COL = 'discretized_action'
    REWARD_COL = 'reward'
    EPISODE_COL = 'icustay_id'
    TIMESTEP_COL = 'timestep'

    LATENT_STATE_DIM = 64
    NUM_ACTIONS = 25
    HIDDEN_DIMS = (128, 128)

    # load training data
    df_train = pd.read_csv('../preprocessing/datasets/mimic-iii/handcrafted/mimic-iii_train_handcrafted.csv', index_col=0)

    # create encoder
    encoder_model = CKCNN(in_channels=len(STATE_SPACE_FEATURES), out_channels=LATENT_STATE_DIM)

    # create DQN controller
    dqn_model = DuelingDQN(state_dim=LATENT_STATE_DIM, num_actions=NUM_ACTIONS, hidden_dims=HIDDEN_DIMS)

    # fit model
    fit_dueling_double_DQN_with_history(model=dqn_model,
                                        encoder=encoder_model,
                                        dataset=df_train,
                                        state_cols=STATE_SPACE_FEATURES,
                                        action_col=ACTION_COL,
                                        reward_col=REWARD_COL,
                                        episode_col=EPISODE_COL,
                                        timestep_col=TIMESTEP_COL,
                                        alpha=1e-4,
                                        gamma=0.9,
                                        tau=1e-3,
                                        num_episodes=4000,
                                        batch_size=32,
                                        eval_after=42,
                                        scheduler_gamma=0.95,
                                        step_scheduler_after=200)
