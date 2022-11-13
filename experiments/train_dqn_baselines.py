"""
Author:   Thomas Bellucci
Filename: train_dqn_with_baselines.py
Descr.:   Performs the training of a Dueling Double DQN model with state-space
          encoder over entire histories.
Date:     01-10-2022
"""

import torch
import pandas as pd
from q_learning import DQN, fit_double_dqn
from baseline_encoders import *
from utils import load_data, count_parameters, Callback


if __name__ == '__main__':
    # Choose history-encoding model and i/o dimensions
    METHOD = 'roggeveen'
    IN_CHANNELS = 48
    OUT_CHANNELS = 48
    BATCH_SIZE = 32

    # Training and validation sets
    train_df = load_data('../preprocessing/datasets/mimic-iii/aggregated_1h/mimic-iii_train.csv')
    valid_df = load_data('../preprocessing/datasets/mimic-iii/aggregated_1h/mimic-iii_valid.csv')

    if METHOD == 'roggeveen':
        encoder = LastState()
    elif METHOD == 'concat-2':
        encoder = StateConcatenation(IN_CHANNELS, OUT_CHANNELS, k=2)
    elif METHOD == 'concat-3':
        encoder = StateConcatenation(IN_CHANNELS, OUT_CHANNELS, k=3)
    elif METHOD == 'causal_cnn':
        encoder = CausalCNN((IN_CHANNELS, OUT_CHANNELS), kernel_sizes=(18,), dilations=(1,))
    elif METHOD == 'lstm':
        encoder = LSTM(IN_CHANNELS, OUT_CHANNELS, batch_size=BATCH_SIZE)
    elif METHOD == 'gru':
        encoder = GRU(IN_CHANNELS, OUT_CHANNELS)
    else:
        raise Exception('Method %s not recognized' % METHOD)
    print('%s parameters: %d' % (METHOD, count_parameters(encoder)))

    dqn = DQN(state_dim=OUT_CHANNELS, hidden_dims=(96, 96), num_actions=25, disallowed_actions=(1, 2, 3, 4))
    print('DQN parameters:  ', count_parameters(dqn))

    callback = Callback(behavior_policy_file='../ope/physician_policy/aggregated_1h/mimic-iii_valid_behavior_policy.csv',
                        valid_data=valid_df)

    fit_double_dqn(experiment='results/%s_experiment' % METHOD,
                   policy=dqn,
                   encoder=encoder,
                   dataset=train_df,
                   lrate=1e-4,
                   gamma=0.9,
                   tau=1e-4,
                   lambda_reward=5,
                   num_episodes=15000,
                   batch_size=BATCH_SIZE,
                   eval_func=callback,
                   eval_after=500,
                   scheduler_gamma=0.95,
                   replay_alpha=0.6,
                   replay_beta=0.9,
                   step_scheduler_after=10000,
                   min_max_reward=(-15, 15))
