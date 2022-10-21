"""
Author:   Thomas Bellucci
Filename: train_dqn_markovian.py
Descr.:   Performs the training of a Dueling Double DQN model as described in
          (Roggeveen et al., 2021) using a handcrafted feature set.
Date:     01-10-2022
"""

import pandas as pd
from q_learning import DQN, fit_double_dqn
from importance_sampling import WIS
from physician import Physician


class OPECallback:
    """ Callback to compute a WIS estimate of V^πe during training
        and the CE loss between model and physician action probabilities
    """
    def __init__(self, behavior_policy_file, valid_data):
        # Load behavior policy that was used to sample validation set
        self._wis = WIS(behavior_policy_file)
        self._phys = Physician(behavior_policy_file)
        self._states = valid_data.filter(regex='x\d+').values

    def __call__(self, policy):
        action_probs = policy.action_probs(self._states)
        return {'wis': self._wis(action_probs),
                'phys_entropy': self._phys(action_probs)}


if __name__ == '__main__':
    # Training and validation data
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv')
    print('train.size = %s  valid.size = %s' % (len(train_df), len(valid_df)))

    # Evaluation callback using OPE
    callback = OPECallback(behavior_policy_file='../ope/physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv',
                           valid_data=valid_df)

    # Optimize DQN model (Note: disallow marks impossible actions)
    dqn_model = DQN(state_dim=48, num_actions=25, hidden_dims=(128, 128), disallow=[1, 2, 3, 4])

    fit_double_dqn(experiment='results/roggeveen_experiment',
                   policy=dqn_model,
                   states=train_df.filter(regex='x\d+'),
                   actions=train_df.action,
                   rewards=train_df.reward,
                   episodes=train_df.episode,  # i.e. icustay_id
                   alpha=1e-4,
                   gamma=0.9,
                   lamda=5,
                   tau=1e-4,
                   num_episodes=30000,
                   batch_size=32,
                   replay_params=(0.4, 0.6),  # was (0.6, 0.9)
                   eval_func=callback,
                   eval_after=1000,
                   scheduler_gamma=0.95,
                   step_scheduler_after=2000,
                   min_max_reward=(-15, 15),
                   lamda_physician=0.0)
