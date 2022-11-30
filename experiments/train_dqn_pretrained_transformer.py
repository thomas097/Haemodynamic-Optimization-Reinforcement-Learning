import torch
import pandas as pd
from q_learning import DQN, fit_double_dqn
from utils import load_data, load_pretrained, count_parameters, OPECallback


if __name__ == '__main__':
    train_df = load_data('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h/train.csv')
    valid_df = load_data('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h/valid.csv')

    encoder = load_pretrained('../pretraining/results/transformer_nsp_pretraining_00000/encoder.pt')
    print('Encoder params:', count_parameters(encoder))

    dqn = DQN(state_dim=32, hidden_dims=(96, 96,), num_actions=25, disallowed_actions=(1, 2, 3, 4))
    print('DQN params:    ', count_parameters(dqn))

    # intermittently evaluate model
    behavior_policy_file = '../ope/physician_policy/amsterdam-umc-db_aggregated_full_cohort_1h_knn/valid_behavior_policy.csv'
    callback = OPECallback(behavior_policy_file=behavior_policy_file, valid_data=valid_df)

    fit_double_dqn(
        experiment='results/pretrained_transformer_frozen_experiment',
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
        freeze_encoder=True, # REMOVE!
        step_scheduler_after=10000,
        min_max_reward=(-100, 100)
    )
