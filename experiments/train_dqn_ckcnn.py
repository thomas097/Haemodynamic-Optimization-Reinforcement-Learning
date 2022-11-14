"""
Author:   Thomas Bellucci
Filename: train_dqn_with_ckcnn.py
Descr.:   Performs the training of a Dueling Double DQN model with state-space
          encoder over entire histories.
Date:     01-10-2022
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from ckcnn import CKCNN
from q_learning import DQN, fit_double_dqn
from experience_replay import EvaluationReplay
from importance_sampling import WeightedIS
from physician_entropy import PhysicianEntropy
from utils import load_data, count_parameters


class OPECallback:
    """
        Callback which evaluates policy π on a validation set of
        states and returns the WIS estimate of V^πe.
    """
    def __init__(self, behavior_policy_file, valid_data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._wis = WeightedIS(behavior_policy_file)
        self._phys = PhysicianEntropy(behavior_policy_file)
        self._replay = EvaluationReplay(valid_data, return_history=True, device=device)

    def __call__(self, encoder, policy, batch_size=128):
        with torch.no_grad():
            encoded_states = torch.concat([encoder(t).detach() for t in tqdm(self._replay.iterate(batch_size))])
            action_probs = policy.action_probs(encoded_states)

        weighted_is = self._wis(action_probs)
        phys_entropy = self._phys(action_probs)
        return {'wis': weighted_is, 'physician_entropy': phys_entropy}


if __name__ == '__main__':
    train_df = load_data('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv')
    valid_df = load_data('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv')

    encoder = CKCNN(layer_channels=(48, 64), max_timesteps=18)
    print('CKCNN parameters:', count_parameters(encoder))

    dqn = DQN(state_dim=64, hidden_dims=(128, 128), num_actions=25, disallowed_actions=(1, 2, 3, 4))
    print('DQN parameters:  ', count_parameters(dqn))

    callback = OPECallback(behavior_policy_file='../ope/physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv',
                           valid_data=valid_df)

    fit_double_dqn(experiment='results/ckcnn_experiment',
                   policy=dqn,
                   encoder=encoder,
                   dataset=train_df,
                   timedelta='4h',
                   lrate=1e-4,
                   gamma=0.9,
                   tau=1e-4,
                   lambda_reward=5,
                   num_episodes=20000,
                   batch_size=32,
                   eval_func=callback,
                   eval_after=500,
                   scheduler_gamma=0.95,
                   replay_alpha=0.6,
                   replay_beta=0.9,
                   step_scheduler_after=10000,
                   min_max_reward=(-15, 15),
                   save_on='physician_entropy')