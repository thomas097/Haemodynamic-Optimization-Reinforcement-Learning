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
from utils import load_data, load_pretrained, count_parameters, Callback


if __name__ == '__main__':
    train_df = load_data('../preprocessing/datasets/mimic-iii/non_aggregated_1h/mimic-iii_train.csv')
    valid_df = load_data('../preprocessing/datasets/mimic-iii/non_aggregated_1h/mimic-iii_valid.csv')
    print('Loaded trainval data')

    encoder = load_pretrained('../pretraining/results/transformer_v2_pretraining_00000/encoder.pt')
    print('Encoder params:', count_parameters(encoder))

    dqn = DQN(state_dim=48, hidden_dims=(96, 96,), num_actions=25, disallowed_actions=(1, 2, 3, 4))
    print('DQN params:    ', count_parameters(dqn))

    callback = OPECallback(behavior_policy_file='../ope/physician_policy/aggregated_1h/mimic-iii_valid_behavior_policy.csv',
                           valid_data=valid_df)

    fit_double_dqn(experiment='results/transformer_experiment',
                   policy=dqn,
                   encoder=encoder,
                   dataset=train_df,
                   lrate=1e-4,
                   gamma=0.9,
                   tau=1e-4,
                   lambda_reward=5,
                   num_episodes=50000,
                   batch_size=8,
                   truncate=256,
                   eval_func=callback,
                   eval_after=250,
                   scheduler_gamma=0.95,
                   replay_alpha=0.6,
                   replay_beta=0.9,
                   step_scheduler_after=10000,
                   min_max_reward=(-15, 15),
                   lambda_consv=0.3,  # Limit bootstrapping from OOD actions
                   freeze_encoder=True,
                   save_on='physician_entropy')
