import pandas as pd
import numpy as np
from sklearn.preprocessing import label_binarize
from importance_sampling import WIS
from fitted_Q_evaluation import FQE


class WDR:
    def __init__(self, behavior_policy_file, q_estimator, gamma=0.9):
        """ Implementation of the Stepwise Weighted Doubly Robust (WDR) estimator.
            Please refer to https://arxiv.org/pdf/1807.01066.pdf for details.

            Params
            behavior_policy_file: DataFrame containing action probabilities (columns '0'-'24') for behavior policy,
                                  chosen actions ('action') and associated rewards ('reward').
            q_estimator:          Estimator of Q-function of πe with a `predict(state_action_table)` method
            gamma:                Discount factor
        """
        self._wis = WIS(behavior_policy_file, gamma)
        self._q = q_estimator
        self._gamma = gamma

    def _wdr_covariates(self, pi):
        # Predict Q-values given state-action table
        Q = self._q.predict(self._q._state_action)
        return -1

    def __call__(self, pi):
        if not self._q._fitted:
            self._q._fit_estimator(pi)
        return self._wis(pi) - self._wdr_covariates(pi)


if __name__ == '__main__':
    # State-space as described in (Roggeveen et al., 2021).
    STATE_COLS = ['max_vp', 'total_iv_fluid', 'sirs_score', 'sofa_score', 'weight', 'ventilator', 'height',
                  'age', 'gender', 'heart_rate', 'temp', 'mean_bp', 'dias_bp', 'sys_bp', 'resp_rate', 'spo2',
                  'natrium', 'chloride', 'kalium', 'trombo', 'leu', 'anion_gap', 'aptt', 'art_ph', 'asat',
                  'alat', 'bicarbonaat', 'art_be', 'ion_ca', 'lactate', 'paco2', 'pao2', 'shock_index', 'hb',
                  'bilirubin', 'creatinine', 'inr', 'ureum', 'albumin', 'magnesium', 'calcium', 'pf_ratio',
                  'glucose', 'running_total_urine_output', 'total_urine_output', 'running_total_iv_fluid']

    TRAINING_DATA = '../../preprocessing/datasets/mimic-iii/roggeveen/mimic-iii_train.csv'

    # Metric
    fqe = FQE(TRAINING_DATA, STATE_COLS, 'action', 'reward', 'icustay_id')

    wdr = WDR('physician_policy/mimic-iii_train_behavior_policy.csv', fqe)

    # Behavior policy
    behavior_df = pd.read_csv('physician_policy/mimic-iii_train_behavior_policy.csv')
    behavior_policy = behavior_df[[str(i) for i in range(25)]].values

    # Random policy
    random_policy = np.random.uniform(0.0, 1, behavior_policy.shape)
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    print('WDR:')
    print('behavior:', wdr(behavior_policy))
    print('random:  ', wdr(random_policy))

    # Sanity check: estimate true discounted reward
    rewards = behavior_df['reward'].values.reshape(-1, 18)[:, :-1]
    gamma = np.power(0.9, np.arange(17))[np.newaxis]
    discounted_reward = np.mean(np.sum(gamma * rewards, axis=1))
    print('\nTrue reward of pi_b:', discounted_reward)