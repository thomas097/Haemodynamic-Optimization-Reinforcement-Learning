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
    # Define columns marking state- and action-space
    STATE_SPACE_FEATURES = ['max_vp', 'total_iv_fluid', 'sirs_score', 'sofa_score', 'weight', 'ventilator', 'height',
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
    HIDDEN_DIMS = (128, 128)  # see (Roggeveen et al., 2021)

    # load training data
    df_train = pd.read_csv('../preprocessing/datasets/mimic-iii/handcrafted/mimic-iii_train_handcrafted.csv', index_col=0)
    print(df_train)

    # create DQN controller
    dqn_model = DuelingDQN(state_dim=len(STATE_SPACE_FEATURES), num_actions=NUM_ACTIONS, hidden_dims=HIDDEN_DIMS)

    # fit model
    fit_dueling_double_DQN(experiment_name='roggeveen_experiment',
                           policy=dqn_model,
                           dataset=df_train,
                           state_cols=STATE_SPACE_FEATURES,
                           action_col=ACTION_COL,
                           reward_col=REWARD_COL,
                           episode_col=EPISODE_COL,
                           timestep_col=TIMESTEP_COL,
                           alpha=1e-4,
                           gamma=0.9,
                           tau=1e-2,
                           num_episodes=4000,
                           batch_size=8,
                           scheduler_gamma=0.95,
                           step_scheduler_after=200,
                           reward_clipping=15)
