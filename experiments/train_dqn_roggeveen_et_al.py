"""
Author:   Thomas Bellucci
Filename: train_dqn_roggeveen_et_al.py
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
    def __init__(self, behavior_policy_file, states):
        # Load behavior policy that was used to sample validation set
        self._wis = WIS(behavior_policy_file)
        self._phys = Physician(behavior_policy_file)
        self._states = states.values

    def __call__(self, policy):
        action_probs = policy.action_probs(self._states)
        return {'wis': self._wis(action_probs),
                'phys_entropy': self._phys(action_probs)}


if __name__ == '__main__':
    # State-space as described in (Roggeveen et al., 2021).
    STATE_COLUMNS = ['max_vaso', 'total_iv_fluid', 'sirs_score', 'sofa_score', 'weight', 'ventilator', 'height', 'age',
                     'gender', 'heart_rate', 'temp', 'mean_bp', 'dias_bp', 'sys_bp', 'resp_rate', 'spo2', 'natrium',
                     'chloride', 'kalium', 'trombo', 'leu', 'anion_gap', 'aptt', 'art_ph', 'asat', 'fio2', 'alat',
                     'bicarbonaat', 'art_be', 'ion_ca', 'lactate', 'paco2', 'pao2', 'shock_index', 'hb', 'bilirubin',
                     'creatinine', 'inr', 'ureum', 'albumin', 'magnesium', 'calcium', 'pf_ratio', 'glucose',
                     'running_total_urine_output', 'total_urine_output', 'running_total_iv_fluid']

    # Training and validation data
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv')
    print('train.size = %s  valid.size = %s' % (len(train_df), len(valid_df)))

    # Evaluation callback using OPE
    callback = OPECallback(behavior_policy_file='../ope/physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv',
                           states=valid_df[STATE_COLUMNS])

    # Optimize DQN model (1-4 correspond to no IV but vasopressor!)
    dqn_model = DQN(state_dim=len(STATE_COLUMNS), num_actions=25, hidden_dims=(128, 128), disallow=[1, 2, 3, 4])

    fit_double_dqn(experiment='roggeveen_experiment',
                   policy=dqn_model,
                   states=train_df[STATE_COLUMNS],
                   actions=train_df.action,
                   rewards=train_df.reward,
                   episodes=train_df.icustay_id,
                   alpha=1e-4,
                   gamma=0.9,
                   lamda=5,
                   tau=1e-4,
                   num_episodes=100000,
                   batch_size=32,
                   replay_params=(0.4, 0.6),  # was (0.6, 0.9)
                   eval_func=callback,
                   eval_after=500,
                   scheduler_gamma=0.95,
                   step_scheduler_after=2000,
                   min_max_reward=(-15, 15),
                   lamda_physician=1)
