# to allow running from terminal without pyCharm :(
import sys

sys.path.append('../ope')
sys.path.append('../models/q_learning')
sys.path.append('../models/attention/nn')
sys.path.append('../models/ckconv/nn')
sys.path.append('../models')

import torch
import argparse
import pandas as pd
from q_learning import DQN, fit_double_dqn
from baseline_encoders import *
from utils import load_data, count_parameters, OPECallback, add_intermediate_reward


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains policy with pretrained encoder')
    parser.add_argument('-d', "--dataset", type=str, default='amsterdam-umc-db')
    parser.add_argument('-m', "--baseline", type=str, default='last_state')
    parser.add_argument('-o', "--out_dims", type=int, default=96)
    parser.add_argument('-a', "--alpha", type=float, default=1e-4)
    parser.add_argument('-g', "--gamma", type=float, default=0.95)
    parser.add_argument('-l', "--tau", type=float, default=1e-4)
    parser.add_argument('-e', "--episodes", type=int, default=200000)
    parser.add_argument('-k', "--eval_after", type=int, default=5000)
    parser.add_argument('-z', "--freeze_encoder", type=bool, default=False)
    parser.add_argument('-w', "--replay_alpha", type=float, default=0.6)
    parser.add_argument('-q', "--replay_beta", type=float, default=0.9)
    parser.add_argument('-b', "--batch_size", type=int, default=32)
    args = vars(parser.parse_args())
    print('Running with args:', args)

    behavior_policy_file = '../ope/physician_policy/%s_aggregated_full_cohort_2h_mlp/valid_behavior_policy.csv' % args['dataset']
    train_df = load_data('../preprocessing/datasets/%s/aggregated_full_cohort_2h/train.csv' % args['dataset'])
    valid_df = load_data('../preprocessing/datasets/%s/aggregated_full_cohort_2h/valid.csv' % args['dataset'])

    # Add intermediate rewards based on MAP and lactate
    train_df.reward = add_intermediate_reward(train_df)
    valid_df.reward = add_intermediate_reward(valid_df)

    in_channels = train_df.filter(regex='x\d+').shape[1]

    if args['baseline'] == 'last_state':
        encoder = LastState(in_channels, args['out_dims'])
    elif args['baseline'] == 'concat-2':
        encoder = StateConcatenation(in_channels, args['out_dims'], k=2)
    elif args['baseline'] == 'concat-3':
        encoder = StateConcatenation(in_channels, args['out_dims'], k=3)
    elif args['baseline'] == 'tcn':
        encoder = TCN((in_channels, 64, args['out_dims']), kernel_sizes=(18, 3), dilations=(2, 1))
    elif args['baseline'] == 'lstm':
        encoder = LSTM(in_channels, args['out_dims'])
    elif args['baseline'] == 'gru':
        encoder = GRU(in_channels, args['out_dims'])
    else:
        raise Exception('Method %s not recognized' % args['baseline'])
    print('%s parameters: %d' % (args['baseline'], count_parameters(encoder)))

    dqn = DQN(state_dim=args['out_dims'], hidden_dims=(96, 96,), num_actions=25, disallowed_actions=(1, 2, 3, 4))
    print('DQN parameters:  ', count_parameters(dqn))

    # intermittently evaluate policy
    callback = OPECallback(behavior_policy_file=behavior_policy_file, valid_data=valid_df)

    # Fit!
    fit_double_dqn(
        experiment='results/%s/%s_experiment' % (args['dataset'], args['baseline']),
        policy=dqn,
        encoder=encoder,
        dataset=train_df,
        lrate=args['alpha'],
        gamma=args['gamma'],
        tau=args['tau'],
        lambda_reward=1,
        num_episodes=args['episodes'],
        batch_size=args['batch_size'],
        freeze_encoder=args['freeze_encoder'],  # set freeze_encoder=True to disable fine-tuning!
        eval_func=callback,
        eval_after=args['eval_after'],
        scheduler_gamma=0.95,
        replay_alpha=args['replay_alpha'],
        replay_beta=args['replay_beta'],
        step_scheduler_after=10000,
        min_max_reward=(-27, 10),
        save_on=None
    )
