"""
Author:   Thomas Bellucci
Filename: DQN_roggeveen_et_al.py
Descr.:   Performs the training of a Dueling Double DQN model with state-space
          encoder over entire histories.
Date:     01-10-2022
"""

import torch
import pandas as pd

from DQN import DuelingDQN, fit_dueling_double_DQN
from ckconv_v3 import CKConv


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
    fit_dueling_double_DQN(experiment_name='CKCNN_experiment',
                           policy=dqn_model,
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
