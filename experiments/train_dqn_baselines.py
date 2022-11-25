import torch
import pandas as pd
from q_learning import DQN, fit_double_dqn
from baseline_encoders import *
from utils import load_data, count_parameters, OPECallback


if __name__ == '__main__':
    # Choose history-encoding model and i/o dimensions
    METHOD = 'last_state'
    IN_CHANNELS = 55
    OUT_CHANNELS = 55

    # Training and validation sets
    train_df = load_data('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h/train.csv')
    valid_df = load_data('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h/valid.csv')

    if METHOD == 'last_state':
        encoder = LastState()
    elif METHOD == 'concat-2':
        encoder = StateConcatenation(IN_CHANNELS, OUT_CHANNELS, k=2)
    elif METHOD == 'concat-3':
        encoder = StateConcatenation(IN_CHANNELS, OUT_CHANNELS, k=3)
    elif METHOD == 'causal_cnn':
        encoder = CausalCNN((IN_CHANNELS, OUT_CHANNELS), kernel_sizes=(18,), dilations=(1,))
    elif METHOD == 'lstm':
        encoder = LSTM(IN_CHANNELS, OUT_CHANNELS)
    elif METHOD == 'gru':
        encoder = GRU(IN_CHANNELS, OUT_CHANNELS)
    else:
        raise Exception('Method %s not recognized' % METHOD)
    print('%s parameters: %d' % (METHOD, count_parameters(encoder)))

    dqn = DQN(state_dim=OUT_CHANNELS, hidden_dims=(96, 96), num_actions=25, disallowed_actions=(1, 2, 3, 4))
    print('DQN parameters:  ', count_parameters(dqn))

    callback = OPECallback(behavior_policy_file='../ope/physician_policy/amsterdam-umc-db_aggregated_full_cohort_1h_knn/valid_behavior_policy.csv',
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
                   batch_size=32,
                   eval_func=callback,
                   eval_after=500,
                   scheduler_gamma=0.95,
                   replay_alpha=0.6,
                   replay_beta=0.9,
                   step_scheduler_after=10000,
                   min_max_reward=(-100, 100))
