"""
Author:   Thomas Bellucci
Filename: DQN_roggeveen_et_al.py
Descr.:   Performs the training of a Dueling Double DQN model as described in
          (Roggeveen et al., 2021) using a handcrafted feature set.
Date:     01-10-2022
"""

import pandas as pd
from DQN import DuelingDQN, fit_dueling_double_DQN


if __name__ == '__main__':
    # State space and hyper-parameters as described in Roggeveen et al.
    STATE_COLS = ['max_vp', 'total_iv_fluid', 'sirs_score', 'sofa_score', 'weight', 'ventilator', 'height',
                  'age', 'gender', 'heart_rate', 'temp', 'mean_bp', 'dias_bp', 'sys_bp', 'resp_rate', 'spo2',
                  'natrium', 'chloride', 'kalium', 'trombo', 'leu', 'anion_gap', 'aptt', 'art_ph', 'asat',
                  'alat', 'bicarbonaat', 'art_be', 'ion_ca', 'lactate', 'paco2', 'pao2', 'shock_index', 'hb',
                  'bilirubin', 'creatinine', 'inr', 'ureum', 'albumin', 'magnesium', 'calcium', 'pf_ratio',
                  'glucose', 'running_total_urine_output', 'total_urine_output', 'running_total_iv_fluid']
    ACTION_COL = 'discretized_action'
    REWARD_COL = 'reward'
    EPISODE_COL = 'icustay_id'
    TIMESTEP_COL = 'timestep'

    NUM_ACTIONS = 25
    HIDDEN_DIMS = (128, 128)

    # load training and validation data
    df_train = pd.read_csv('../preprocessing/datasets/mimic-iii/handcrafted/mimic-iii_train_handcrafted.csv')
    df_valid = pd.read_csv('../preprocessing/datasets/mimic-iii/handcrafted/mimic-iii_valid_handcrafted.csv')
    print('train.size = %s  valid.size = %s' % (len(df_train), len(df_valid)))

    # DQN instance
    dqn_model = DuelingDQN(state_dim=len(STATE_COLS), num_actions=NUM_ACTIONS, hidden_dims=HIDDEN_DIMS)

    # optimize policy
    fit_dueling_double_DQN(experiment_name='roggeveen_experiment',
                           policy=dqn_model,
                           dataset=df_train,
                           state_cols=STATE_COLS,
                           action_col=ACTION_COL,
                           reward_col=REWARD_COL,
                           episode_col=EPISODE_COL,
                           timestep_col=TIMESTEP_COL,
                           alpha=1e-4,
                           gamma=0.9,
                           tau=1e-2,
                           num_episodes=4000,
                           batch_size=8,
                           eval_after=100,
                           scheduler_gamma=0.95,
                           step_scheduler_after=200,
                           reward_clipping=15)
