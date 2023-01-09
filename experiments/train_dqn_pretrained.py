import torch
import pandas as pd
from q_learning import DQN, fit_double_dqn
from utils import load_data, load_pretrained, count_parameters, OPECallback, add_intermediate_reward


if __name__ == '__main__':
    behavior_policy_file = '../ope/physician_policy/amsterdam-umc-db_aggregated_full_cohort_2h_mlp/valid_behavior_policy.csv'
    train_df = load_data('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_2h/train.csv')
    valid_df = load_data('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_2h/valid.csv')

    # Add intermediate rewards based on MAP and lactate
    train_df.reward = add_intermediate_reward(train_df)
    valid_df.reward = add_intermediate_reward(valid_df)

    encoder = load_pretrained('../pretraining/results/amsterdam-umc-db/transformer_mt_pretraining_00000/encoder.pt')
    print('Encoder params:', count_parameters(encoder))

    dqn = DQN(state_dim=96, hidden_dims=(96, 96), num_actions=25, disallowed_actions=(1, 2, 3, 4))
    print('DQN params:    ', count_parameters(dqn))

    # intermittently evaluate model
    callback = OPECallback(behavior_policy_file=behavior_policy_file, valid_data=valid_df)

    fit_double_dqn(
        experiment='results/amsterdam-umc-db/latent_dims_results/ckcnn_128dims_pretraining',
        policy=dqn,
        encoder=encoder,
        dataset=train_df,
        lrate=1e-4,
        gamma=0.95,
        tau=1e-4,
        lambda_reward=1,
        num_episodes=200000,
        batch_size=32,
        freeze_encoder=False, # set freeze_encoder=True to disable fine-tuning!
        eval_func=callback,
        eval_after=5000,
        scheduler_gamma=0.95,
        replay_alpha=0.6,
        replay_beta=0.9,
        step_scheduler_after=10000,
        min_max_reward=(-27, 10),
        save_on=None
    )
