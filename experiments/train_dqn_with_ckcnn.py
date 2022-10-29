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

from q_learning import DQN, fit_double_dqn
from experience_replay import EvaluationReplay
from ckcnn import CKCNN
from importance_sampling import WeightedIS
from physician_entropy import PhysicianEntropy


class OPECallback:
    """
        Callback which evaluates policy π on a validation set of
        states and returns the WIS estimate of V^πe.
    """
    def __init__(self, behavior_policy_file, valid_data):
        # Load validation set and metrics
        self._wis = WeightedIS(behavior_policy_file, min_samples=20)
        self._phys = PhysicianEntropy(behavior_policy_file)
        self._replay = EvaluationReplay(valid_data, return_history=True)

    def __call__(self, encoder, policy):
        # Feed histories through encoder to get fixed state representation
        encoded_states = torch.concat([encoder(t) for t in self._replay.iterate()])

        # Action probs from state vectors
        action_probs = policy.action_probs(encoded_states)

        actions, counts = np.unique(np.argmax(action_probs, axis=1), return_counts=True)
        i = np.argmax(counts)
        print('\nmost_common_action:', actions[i])
        print('p_most_common_action:', counts[i] / np.sum(counts))

        # Metrics
        weighted_is = self._wis(action_probs)
        phys_entropy = self._phys(action_probs)
        return {'wis': weighted_is, 'phys_entropy': phys_entropy}


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Load training and validation data
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h_with_cv/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h_with_cv/mimic-iii_valid.csv')
    print('train.size = %s  valid.size = %s' % (len(train_df) // 18, len(valid_df) // 18))

    # Setup encoder model
    encoder = CKCNN(layer_channels=(48, 64), max_timesteps=18)
    print('CKCNN parameters:', count_parameters(encoder))

    # Create Dueling DQN controller
    dqn = DQN(state_dim=64, hidden_dims=(128,), num_actions=25)
    print('DQN parameters:  ', count_parameters(dqn))

    # Handles intermittent evaluation using OPE on validation set
    callback = OPECallback(behavior_policy_file='../ope/physician_policy/roggeveen_4h_with_cv/mimic-iii_valid_behavior_policy.csv',
                           valid_data=valid_df)
    # Fit model
    fit_double_dqn(experiment='results/ckcnn_experiment',
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
                   eval_after=500,
                   scheduler_gamma=0.95,
                   replay_alpha=0.6,
                   replay_beta=0.9,
                   step_scheduler_after=10000,
                   min_max_reward=(-15, 15))
