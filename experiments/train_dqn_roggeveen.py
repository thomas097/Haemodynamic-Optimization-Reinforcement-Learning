"""
Author:   Thomas Bellucci
Filename: train_dqn_roggeveen.py
Descr.:   Performs the training of a Dueling Double DQN model as described in
          (Roggeveen et al., 2021) using a handcrafted feature set and assuming
          a Markovian state space.
Date:     01-10-2022
"""

import numpy as np
import pandas as pd
import torch

from q_learning import DQN, fit_double_dqn
from experience_replay import EvaluationReplay
from importance_sampling import WeightedIS, IS
from physician_entropy import PhysicianEntropy
from utils import load_data, count_parameters


class OPECallback:
    """
        Callback to compute a WIS estimate of V^πe during training
        and the entropy between model and physician action probabilities
    """
    def __init__(self, behavior_policy_file, valid_data):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load behavior policy that was used to generate validation set
        self._wis = WeightedIS(behavior_policy_file, verbose=True)
        self._phys = PhysicianEntropy(behavior_policy_file)
        self._replay = EvaluationReplay(valid_data, return_history=False, device=device)  # Return Markov states!

    def __call__(self, policy, batch_size=128):
        with torch.no_grad():
            encoded_states = torch.concat([t.detach() for t in self._replay.iterate(batch_size)])
            action_probs = policy.action_probs(encoded_states)

        weighted_is = self._wis(action_probs)
        phys_entropy = self._phys(action_probs)
        return {'wis': weighted_is, 'physician_entropy': phys_entropy}


if __name__ == '__main__':
    train_df = load_data('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv')
    valid_df = load_data('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv')

    callback = OPECallback(behavior_policy_file='../ope/physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv',
                           valid_data=valid_df)

    dqn_model = DQN(state_dim=48, num_actions=25, hidden_dims=(128, 128), disallowed_actions=(1, 2, 3, 4))

    fit_double_dqn(experiment='results/roggeveen_experiment',
                   policy=dqn_model,
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
                   min_max_reward=(-15, 15),
                   #lambda_consv=0.2,  # Limit bootstrapping from OOD actions
                   save_on=None)
