"""
Author:   Thomas Bellucci
Filename: train_dqn_with_ckcnn.py
Descr.:   Performs the training of a Dueling Double DQN model with state-space
          encoder over entire histories.
Date:     01-10-2022
"""

import torch
import pandas as pd

from q_learning import DQN, fit_double_dqn
from ckcnn import CKCNN
from importance_sampling import WIS


class OPECallback:
    """ Callback which evaluates policy π on a validation set of
        states and returns the WIS estimate of V^πe.
    """
    def __init__(self, behavior_policy_file, validation_df):
        # Load behavior policy that was used to sample validation set
        self._estimator = WIS(behavior_policy_file)
        self._valid_df = validation_df

    def _generate_histories(self):
        for i, row in self._valid_df.iterrows():
            # Extract history
            ep = row['icustay_id']
            history = self._valid_df[(self._valid_df['icustay_id'] == ep) & (self._valid_df.index <= i + 1)]

            # Drop episode column and create batch dimension
            history = history.loc[:, history.columns != 'icustay_id']
            yield torch.Tensor(history.values).unsqueeze(0)

    def __call__(self, encoder, policy):
        # Generate histories and feed through encoder to get fixed state representation
        states = torch.concat([encoder(t) for t in self._generate_histories()])

        # Compute action probs from state vectors
        action_probs = policy.action_probs(states)
        wis = self._estimator(action_probs)
        return {'wis': wis}


if __name__ == '__main__':
    # Define columns marking state- and action-space
    STATE_COLUMNS = ['max_vp', 'total_iv_fluid', 'sirs_score', 'sofa_score', 'weight', 'ventilator', 'height',
                     'age', 'gender', 'heart_rate', 'temp', 'mean_bp', 'dias_bp', 'sys_bp', 'resp_rate', 'spo2',
                     'natrium', 'chloride', 'kalium', 'trombo', 'leu', 'anion_gap', 'aptt', 'art_ph', 'asat',
                     'alat', 'bicarbonaat', 'art_be', 'ion_ca', 'lactate', 'paco2', 'pao2', 'hb', 'bilirubin',
                     'creatinine', 'inr', 'ureum', 'albumin', 'magnesium', 'calcium', 'glucose', 'total_urine_output']

    # Training and validation data
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen/mimic-iii_valid.csv')
    print('train.size = %s  valid.size = %s' % (len(train_df), len(valid_df)))

    # setup encoder model
    encoder_model = CKCNN(in_channels=len(STATE_COLUMNS), out_channels=64)

    # create Dueling DQN controller
    dqn_model = DQN(state_dim=64, hidden_dims=(128, 128), num_actions=25)

    # Evaluation callback using OPE
    ope_callback = OPECallback(behavior_policy_file='../ope/physician_policy/mimic-iii_valid_behavior_policy.csv',
                               validation_df=valid_df[STATE_COLUMNS + ['icustay_id']])

    # fit model
    fit_double_dqn(experiment='ckcnn_experiment',
                   policy=dqn_model,
                   encoder=encoder_model,
                   states=train_df[STATE_COLUMNS],
                   actions=train_df.action,
                   rewards=train_df.reward,
                   episodes=train_df.icustay_id,
                   timesteps=train_df.timestep,
                   alpha=1e-4,
                   gamma=0.9,
                   tau=1e-3,
                   num_episodes=4000,
                   batch_size=32,
                   eval_func=ope_callback,
                   eval_after=42,
                   scheduler_gamma=0.95,
                   step_scheduler_after=200)
