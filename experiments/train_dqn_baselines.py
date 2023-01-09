import torch
import pandas as pd
from q_learning import DQN, fit_double_dqn
from baseline_encoders import *
from utils import load_data, count_parameters, OPECallback, add_intermediate_reward


if __name__ == '__main__':
    # Choose history-encoding model and output dimensions
    METHOD = 'last_state'  # handcrafted state-space baseline
    OUT_CHANNELS = 96
    HIDDEN_CHANNELS = 32

    # Training and validation sets
    behavior_policy_file = '../ope/physician_policy/amsterdam-umc-db_aggregated_full_cohort_2h_mlp/valid_behavior_policy.csv'
    train_df = load_data('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_2h/train.csv')
    valid_df = load_data('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_2h/valid.csv')

    # Add intermediate rewards based on MAP
    train_df.reward = add_intermediate_reward(train_df, scale=1)
    valid_df.reward = add_intermediate_reward(valid_df, scale=1)

    in_channels = train_df.filter(regex='x\d+').shape[1]

    if METHOD == 'last_state':
        encoder = LastState(in_channels, OUT_CHANNELS)
    elif METHOD == 'concat-2':
        encoder = StateConcatenation(in_channels, OUT_CHANNELS, k=2)
    elif METHOD == 'concat-3':
        encoder = StateConcatenation(in_channels, OUT_CHANNELS, k=3)
    elif METHOD == 'tcn':
        encoder = TCN((in_channels, HIDDEN_CHANNELS, OUT_CHANNELS), kernel_sizes=(18, 3), dilations=(2, 1))
    elif METHOD == 'lstm':
        encoder = LSTM(in_channels, OUT_CHANNELS)
    elif METHOD == 'gru':
        encoder = GRU(in_channels, OUT_CHANNELS)
    else:
        raise Exception('Method %s not recognized' % METHOD)
    print('%s parameters: %d' % (METHOD, count_parameters(encoder)))

    dqn = DQN(state_dim=OUT_CHANNELS, hidden_dims=(96, 96,), num_actions=25, disallowed_actions=(1, 2, 3, 4))
    print('DQN parameters:  ', count_parameters(dqn))

    # intermittently evaluate policy
    callback = OPECallback(behavior_policy_file=behavior_policy_file, valid_data=valid_df)

    # Fit!
    fit_double_dqn(
        experiment='results/%s_experiment' % METHOD,
        policy=dqn,
        encoder=encoder,
        dataset=train_df,
        lrate=1e-4,
        gamma=0.95,
        tau=1e-4,
        lambda_reward=1,
        num_episodes=200000,
        batch_size=32,
        eval_func=callback,
        eval_after=5000,
        scheduler_gamma=0.95,
        replay_alpha=0.6,
        replay_beta=0.9,
        step_scheduler_after=10000,
        min_max_reward=(-27, 10),
        save_on=None
    )
