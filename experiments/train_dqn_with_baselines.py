"""
Author:   Thomas Bellucci
Filename: train_dqn_with_baselines.py
Descr.:   Performs the training of a Dueling Double DQN model with state-space
          encoder over entire histories.
Date:     01-10-2022
"""

import re
import torch
import pandas as pd
from q_learning import DQN, fit_double_dqn
from experience_replay import EvaluationReplay
from baseline_encoders import *
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
        encoded_states = torch.concat([encoder(t) for t in self._replay.iterate(BATCH_SIZE)])

        # Compute action probs from state vectors
        action_probs = policy.action_probs(encoded_states)
        return {'wis': self._wis(action_probs),
                'phys_entropy': self._phys(action_probs)}


if __name__ == '__main__':
    ENCODER_METHOD = 'lstm'
    IN_CHANNELS = 48
    OUT_CHANNELS = 48
    BATCH_SIZE = 32

    # Training and validation data
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv')
    print('train.size = %s  valid.size = %s' % (len(train_df) // 18, len(valid_df) // 18))

    # Choose encoder model
    if ENCODER_METHOD == 'concat-1':  # HACK: Used for debugging!
        encoder = StateConcatenation(k=1)
    elif ENCODER_METHOD == 'concat-2':
        encoder = StateConcatenation(k=2)
    elif ENCODER_METHOD == 'concat-3':
        encoder = StateConcatenation(k=3)
    elif ENCODER_METHOD == 'causal_cnn':
        encoder = CausalCNN((IN_CHANNELS, OUT_CHANNELS), kernel_sizes=(5,), dilations=(2,))
    elif ENCODER_METHOD == 'lstm':
        encoder = LSTM(IN_CHANNELS, OUT_CHANNELS, batch_size=BATCH_SIZE)
    elif ENCODER_METHOD == 'gru':
        encoder = GRU(IN_CHANNELS, OUT_CHANNELS)
    else:
        raise Exception('Method %s not recognized' % ENCODER_METHOD)

    # create Dueling DQN controller
    dqn_model = DQN(state_dim=OUT_CHANNELS, hidden_dims=(128, 128), num_actions=25)

    # evaluation callback using OPE
    callback = OPECallback(behavior_policy_file='../ope/physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv',
                           valid_data=valid_df)

    # fit model
    fit_double_dqn(experiment='results/%s_experiment' % ENCODER_METHOD,
                   policy=dqn_model,
                   encoder=encoder,
                   dataset=train_df,
                   alpha=1e-4,
                   gamma=0.9,
                   lamda=5,
                   tau=1e-4,
                   num_episodes=30000,
                   batch_size=BATCH_SIZE,
                   replay_params=(0.4, 0.6),  # was (0.6, 0.9)
                   eval_func=callback,
                   eval_after=500,
                   scheduler_gamma=0.95,
                   step_scheduler_after=2000,
                   min_max_reward=(-15, 15),
                   lamda_physician=0.0)
