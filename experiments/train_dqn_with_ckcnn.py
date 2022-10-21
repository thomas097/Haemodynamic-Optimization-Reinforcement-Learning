"""
Author:   Thomas Bellucci
Filename: train_dqn_with_ckcnn.py
Descr.:   Performs the training of a Dueling Double DQN model with state-space
          encoder over entire histories.
Date:     01-10-2022
"""

import pandas as pd

import torch
from q_learning import DQN, fit_double_dqn
from experience_replay import EvaluationReplay
from ckcnn import CKCNN
from importance_sampling import WeightedIS
from physician import Physician


class OPECallback:
    """ Callback which evaluates policy π on a validation set of
        states and returns the WIS estimate of V^πe.
    """
    def __init__(self, behavior_policy_file, valid_data):
        # Load behavior policy that was used to sample validation set
        self._wis = WeightedIS(behavior_policy_file)
        self._phys = Physician(behavior_policy_file)
        self._replay = EvaluationReplay(valid_data, return_history=True)

    def __call__(self, encoder, policy):
        # Feed histories through encoder to get fixed state representation
        encoded_states = torch.concat([encoder(t) for t in self._replay.iterate()])

        # Compute action probs from state vectors
        action_probs = policy.action_probs(encoded_states)
        return {'wis': self._wis(action_probs),
                'phys_entropy': self._phys(action_probs)}


if __name__ == '__main__':
    # Training and validation data
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv')
    print('train.size = %s  valid.size = %s' % (len(train_df) // 18, len(valid_df) // 18))

    # setup encoder model
    encoder_model = CKCNN(in_channels=48, out_channels=64)

    # create Dueling DQN controller
    dqn_model = DQN(state_dim=64, hidden_dims=(128,), num_actions=25)

    # evaluation callback using OPE
    callback = OPECallback(behavior_policy_file='../ope/physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv',
                           valid_data=valid_df)

    # fit model
    fit_double_dqn(experiment='results/ckcnn_experiment',
                   policy=dqn_model,
                   encoder=encoder_model,
                   dataset=train_df,
                   alpha=1e-3,
                   gamma=0.9,
                   lamda=5,
                   tau=1e-4,
                   num_episodes=10000,
                   batch_size=32,
                   eval_func=callback,
                   eval_after=1000,
                   scheduler_gamma=0.95,
                   step_scheduler_after=200)
