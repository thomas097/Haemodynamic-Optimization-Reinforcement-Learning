"""
Author:   Thomas Bellucci
Filename: train_dqn_with_ckcnn.py
Descr.:   Performs the training of a Dueling Double DQN model with state-space
          encoder over entire histories.
Date:     01-10-2022
"""

import torch
import pandas as pd
from ckcnn import CKCNN
from q_learning import DQN, fit_double_dqn
from utils import load_data, load_pretrained, count_parameters, OPECallback


if __name__ == '__main__':
    train_df = load_data('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_2h/train.csv')
    valid_df = load_data('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_2h/valid.csv')

    # determine number of clinical measurements, i.e. x* columns
    n_inputs = train_df.filter(regex='x\d+').shape[1]

    # Set up encoder and policy network
    ckcnn = CKCNN(
        layer_channels=(n_inputs, 32),
        d_kernel=24,
        kernel_type='siren',
        max_timesteps=32,
        use_residuals=True,
        fourier_input=True,
    )
    print('CKCNN parameters:', count_parameters(ckcnn))

    dqn = DQN(state_dim=32, hidden_dims=(96, 96), num_actions=25, disallowed_actions=(1, 2, 3, 4))
    print('DQN parameters:  ', count_parameters(dqn))

    # intermittently evaluate model
    behavior_policy_file = '../ope/physician_policy/amsterdam-umc-db_aggregated_full_cohort_2h_knn/valid_behavior_policy.csv'
    callback = OPECallback(behavior_policy_file=behavior_policy_file, valid_data=valid_df)

    fit_double_dqn(
        experiment='results/ckcnn_experiment',
        policy=dqn,
        encoder=ckcnn,
        dataset=train_df,
        lrate=1e-4,
        gamma=0.9,
        tau=1e-4,
        lambda_reward=5,
        lambda_phys=0.8,
        num_episodes=25000,
        batch_size=32,
        eval_func=callback,
        eval_after=500,
        scheduler_gamma=0.95,
        replay_alpha=0.6,
        replay_beta=0.9,
        step_scheduler_after=10000,
        min_max_reward=(-100, 100)
    )
