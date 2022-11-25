import torch
import pandas as pd
from q_learning import DQN, fit_double_dqn
from loss_functions import ProbRegularizer
from utils import load_data, load_pretrained, count_parameters, OPECallback


if __name__ == '__main__':
    train_df = load_data('../preprocessing/datasets/mimic-iii/aggregated_all_1h/mimic-iii_train.csv')
    valid_df = load_data('../preprocessing/datasets/mimic-iii/aggregated_all_1h/mimic-iii_valid.csv')

    encoder = load_pretrained('../pretraining/results/transformer_nsp_pretraining_00001/encoder.pt')
    print('Encoder params:', count_parameters(encoder))

    dqn = DQN(state_dim=16, hidden_dims=(96, 96,), num_actions=25, disallowed_actions=(1, 2, 3, 4))
    print('DQN params:    ', count_parameters(dqn))

    # used to intermittently evaluate model
    callback = OPECallback(behavior_policy_file='../ope/physician_policy/aggregated_all_1h/mimic-iii_valid_behavior_policy.csv',
                           valid_data=valid_df)

    regularizer = ProbRegularizer(behavior_policy_file='../ope/physician_policy/aggregated_all_1h/mimic-iii_train_behavior_policy.csv')

    fit_double_dqn(
        experiment='results/transformer_experiment',
        policy=dqn,
        encoder=encoder,
        dataset=train_df,
        lrate=1e-4,
        gamma=0.9,
        tau=1e-4,
        lambda_reward=5,
        num_episodes=50000,
        batch_size=8,
        eval_func=callback,
        eval_after=250,
        scheduler_gamma=0.95,
        replay_alpha=0.6,
        replay_beta=0.9,
        step_scheduler_after=10000,
        min_max_reward=(-15, 15),
        lambda_reg=0.5,
        #regularizer=regularizer,
        freeze_encoder=True,
        save_on='physician_entropy'
    )
