"""
Author:   Thomas Bellucci
Filename: train_dqn_with_transformer.py
Descr.:   Performs the training of a Dueling Double DQN model with state-space
          encoder over entire histories using the CausalTransformer.
Date:     01-10-2022
"""

import torch
import pandas as pd
from q_learning import DQN, fit_double_dqn
from attention_models import CausalTransformer
from train_dqn_with_ckcnn import OPECallback
from utils import load_data, count_parameters


if __name__ == '__main__':
    train_df = load_data('../preprocessing/datasets/mimic-iii/attention_4h_with_cv/mimic-iii_train.csv')
    valid_df = load_data('../preprocessing/datasets/mimic-iii/attention_4h_with_cv/mimic-iii_valid.csv')

    encoder = CausalTransformer(vocab_size=45, d_model=24, out_channels=64, nheads=1, truncate=256)
    print('Encoder params:', count_parameters(encoder))

    dqn = DQN(state_dim=64, hidden_dims=(128,), num_actions=25)
    print('DQN params:    ', count_parameters(dqn))

    callback = OPECallback(behavior_policy_file='../ope/physician_policy/roggeveen_4h_with_cv/mimic-iii_valid_behavior_policy.csv',
                           valid_data=valid_df)

    fit_double_dqn(experiment='results/transformer_experiment',
                   policy=dqn,
                   encoder=encoder,
                   dataset=train_df,
                   timedelta='4h',
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
                   min_max_reward=(-15, 15))
