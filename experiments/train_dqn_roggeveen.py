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
from importance_sampling import WeightedIS, IS
from physician_entropy import PhysicianEntropy


class OPECallback:
    """
        Callback to compute a WIS estimate of V^πe during training
        and the entropy between model and physician action probabilities
    """
    def __init__(self, behavior_policy_file, valid_data):
        # Load behavior policy that was used to generate validation set
        self._wis = WeightedIS(behavior_policy_file, min_samples=20)
        self._phys = PhysicianEntropy(behavior_policy_file)
        self._states = valid_data.filter(regex='x\d+').values

    def __call__(self, policy):
        policy.eval()
        with torch.no_grad():
            action_probs = policy.action_probs(self._states)
        policy.train()

        actions, counts = np.unique(np.argmax(action_probs, axis=1), return_counts=True)
        i = np.argmax(counts)
        print('\nmost_common_action:', actions[i])
        print('p_most_common_action:', counts[i] / np.sum(counts))

        return {'wis': self._wis(action_probs),
                'phys_entropy': self._phys(action_probs)}


if __name__ == '__main__':
    # Training and validation data
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h_with_cv/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h_with_cv/mimic-iii_valid.csv')
    print('train.size = %s  valid.size = %s' % (len(train_df) // 18, len(valid_df) // 18))

    # Evaluation callback using OPE
    callback = OPECallback(behavior_policy_file='../ope/physician_policy/roggeveen_4h_with_cv/mimic-iii_valid_behavior_policy.csv',
                           valid_data=valid_df)

    # Optimize DQN model
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
                   save_on=False)
