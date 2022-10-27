"""
Author:   Thomas Bellucci
Filename: train_dqn_with_baselines.py
Descr.:   Performs the training of a Dueling Double DQN model with state-space
          encoder over entire histories.
Date:     01-10-2022
"""

import re
import torch
import pandas as pd
from q_learning import DQN, fit_double_dqn
from baseline_encoders import *
from train_dqn_with_ckcnn import OPECallback  # simply reuse callback of the CKCNN


if __name__ == '__main__':
    ENCODER_METHOD = 'gru'
    IN_CHANNELS = 48
    OUT_CHANNELS = 48
    BATCH_SIZE = 32

    # Training and validation data
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv')
    print('train.size = %s  valid.size = %s' % (len(train_df) // 18, len(valid_df) // 18))

    # Choose encoder model
    if ENCODER_METHOD == 'concat-1':  # HACK: Used for debugging!
        encoder = StateConcatenation(k=1)
    elif ENCODER_METHOD == 'concat-2':
        encoder = StateConcatenation(k=2)
    elif ENCODER_METHOD == 'concat-3':
        encoder = StateConcatenation(k=3)
    elif ENCODER_METHOD == 'causal_cnn':
        encoder = CausalCNN((IN_CHANNELS, OUT_CHANNELS), kernel_sizes=(17,), dilations=(1,))
    elif ENCODER_METHOD == 'lstm':
        encoder = LSTM(IN_CHANNELS, OUT_CHANNELS, batch_size=BATCH_SIZE)
    elif ENCODER_METHOD == 'gru':
        encoder = GRU(IN_CHANNELS, OUT_CHANNELS)
    else:
        raise Exception('Method %s not recognized' % ENCODER_METHOD)

    # create Dueling DQN controller
    dqn_model = DQN(state_dim=OUT_CHANNELS, hidden_dims=(128,), num_actions=25)

    # evaluation callback using OPE
    callback = OPECallback(behavior_policy_file='../ope/physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv',
                           valid_data=valid_df)

    # fit model
    fit_double_dqn(experiment='results/%s_experiment' % ENCODER_METHOD,
                   policy=dqn_model,
                   encoder=encoder,
                   dataset=train_df,
                   dt='4h',  # Time between `t` and `t + 1`
                   lrate=1e-4,
                   gamma=0.9,
                   tau=1e-4,
                   lambda_reward=5,
                   num_episodes=50000,
                   batch_size=32,
                   eval_func=callback,
                   eval_after=250,
                   scheduler_gamma=0.95,
                   step_scheduler_after=10000,
                   min_max_reward=(-15, 15),
                   lambda_consv=0.5,
                   save_on='wis')
