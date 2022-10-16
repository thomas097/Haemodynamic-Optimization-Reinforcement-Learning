"""
Author:   Thomas Bellucci
Filename: train_dqn_with_ckcnn.py
Descr.:   Performs the training of a Dueling Double DQN model with state-space
          encoder over entire histories.
Date:     01-10-2022
"""

import pandas as pd

from tqdm import tqdm
from q_learning import DQN, fit_double_dqn
from experience_replay import EvaluationReplay
from ckcnn import CKCNN
from baseline_encoders import *
from importance_sampling import WIS


class OPECallback:
    """ Callback which evaluates policy π on a validation set of
        states and returns the WIS estimate of V^πe.
    """
    def __init__(self, behavior_policy_file, states, episodes):
        # Load behavior policy that was used to sample validation set
        self._wis = WIS(behavior_policy_file)
        self._replay = EvaluationReplay(states, episodes, return_history=True)

    def __call__(self, encoder, policy):
        # Feed histories through encoder to get fixed state representation
        encoded_states = torch.concat([encoder(t) for t in self._replay.iterate()])

        # Compute action probs from state vectors
        action_probs = policy.action_probs(encoded_states)
        return {'wis': self._wis(action_probs)}


if __name__ == '__main__':
    # Define columns marking state- and action-space
    STATE_COLUMNS = ['max_vp', 'total_iv_fluid', 'sirs_score', 'sofa_score', 'weight', 'ventilator', 'height', 'age',
                     'gender', 'heart_rate', 'temp', 'mean_bp', 'dias_bp', 'sys_bp', 'resp_rate', 'spo2', 'natrium',
                     'chloride', 'kalium', 'trombo', 'leu', 'anion_gap', 'aptt', 'art_ph', 'asat', 'fio2', 'alat',
                     'bicarbonaat', 'art_be', 'ion_ca', 'lactate', 'paco2', 'pao2', 'shock_index', 'hb', 'bilirubin',
                     'creatinine', 'inr', 'ureum', 'albumin', 'magnesium', 'calcium', 'pf_ratio', 'glucose',
                     'total_urine_output']

    # Training and validation data
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv')
    print('train.size = %s  valid.size = %s' % (len(train_df), len(valid_df)))

    # setup encoder model
    encoder_model = CKCNN(in_channels=len(STATE_COLUMNS), out_channels=64)

    # create Dueling DQN controller
    dqn_model = DQN(state_dim=64, hidden_dims=(128, 128), num_actions=25)

    # Evaluation callback using OPE
    ope_callback = OPECallback(behavior_policy_file='../ope/physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv',
                               states=valid_df[STATE_COLUMNS], episodes=valid_df['icustay_id'])

    # fit model
    fit_double_dqn(experiment='ckcnn_experiment',
                   policy=dqn_model,
                   encoder=encoder_model,
                   states=train_df[STATE_COLUMNS],
                   actions=train_df.action,
                   rewards=train_df.reward,
                   episodes=train_df.icustay_id,
                   alpha=1e-4,
                   gamma=0.9,
                   tau=1e-3,
                   num_episodes=4000,
                   batch_size=32,
                   eval_func=ope_callback,
                   eval_after=100,
                   scheduler_gamma=0.95,
                   step_scheduler_after=200)
