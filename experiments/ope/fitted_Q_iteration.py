import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso


class FittedQIteration:
    def __init__(self, training_data, state_cols, action_col, reward_col, episode_col, method='rf', params={},
                 max_iters=100, gamma=0.9):
        """ Implementation of Fitted Q-Iteration (FQI) for OPE.
            Please refer to https://arxiv.org/pdf/2107.11003.pdf for details.

            training_file:        DataFrame containing state features and integer actions used to train FQE model
            state_cols:           List of columns corresponding to state-space features
            action_col:           Column corresponding to action chosen
            reward_col:           Column corresponding to reward received
            episode_col:          Column with episode identifiers

            method:  Estimator method to use ('rf'|'lr', default: 'rf')
            params:  Hyperparameters of method (see Sklearn docs)
            gamma:   Dicount factor
        """
        # Import dataset
        df = pd.read_csv(training_data, usecols=state_cols + [action_col, reward_col, episode_col])

        # Get non-terminal 'start states'
        self._states = df[df[reward_col].notna()][state_cols].values

        # Pair rewards and actions with next state (needed for Q-update)
        df[action_col] = df.groupby(episode_col)[action_col].shift(1)
        df[reward_col] = df.groupby(episode_col)[reward_col].shift(1)

        # Drop 'start states' as they aren't 'next states' (with actions and rewards)
        df_next = df[df[action_col].notna()]

        # Extract next states (X_next), actions (y) and rewards (r)
        self._actions = df_next[action_col].values
        self._rewards = df_next[reward_col].values
        self._next_states = df_next[state_cols].values

        # To mask terminal states after reward was issued
        self._is_terminal = (self._rewards != 0)

        # Hyperparameters and model inputs
        self._state_action = self._create_state_action_table()
        self._next_state_action = self._create_next_state_action_table()

        self._max_iters = max_iters
        self._gamma = gamma

        # Estimators
        if method == 'rf':
            self._estimator = RandomForestRegressor(params['n_estimators'], max_depth=params['max_depth'], random_state=42)
        elif method == 'lr':
            self._estimator = Lasso(alpha=params['alpha'])
        else:
            raise Exception('Method %s not understood' % method)
        self.fit()

    def _create_state_action_table(self):
        """ Creates a table of states and corresponding (one-hot encoded) actions in training set.
        """
        num_actions = int(np.max(self._actions) + 1)

        # Concatenate one-hot encoded actions to start states
        one_hot_actions = label_binarize(self._actions, classes=np.arange(num_actions))
        return np.column_stack([self._states, one_hot_actions])

    def _create_next_state_action_table(self):
        """ Creates a table of next states and all (one-hot encoded) actions in action space.
        """
        num_next_states, state_dims = self._next_states.shape
        num_actions = int(np.max(self._actions) + 1)

        # Create table of all possible state-action combinations
        table = np.column_stack([np.repeat(self._next_states, repeats=num_actions, axis=0),
                                 np.tile(np.eye(num_actions), reps=(num_next_states, 1)),
                                 np.repeat(np.arange(num_next_states), repeats=num_actions)])

        # Convert to pandas DataFrame
        header = list(range(num_actions + state_dims)) + ['next_state']
        return pd.DataFrame(data=table, columns=header)

    def fit(self):
        """ Iteratively fits Q-estimator to data.
        """
        # Record metrics over time
        td_errors, meanQs, varQs = [], [], []

        # Initial fit setting Q(s, a) = r
        self._estimator.fit(self._state_action, self._rewards)

        # Drop identifier from next-state-next-action table
        next_state_action = self._next_state_action.values[:, :-1]

        # Perform Q-iteration
        for i in range(self._max_iters):
            # Initial guess of Q(s, a) by the estimator
            Q_prev = self._estimator.predict(self._state_action)

            # Predict Q(s', a') for every state s' and action a'
            self._next_state_action['Q_next'] = self._estimator.predict(next_state_action)

            # Recompute Q(s, a) = r + gamma * max_a'[Q(s', a')]
            max_Q_next = self._next_state_action.groupby('next_state', sort=False)['Q_next'].max()
            Q_target = self._rewards + self._gamma * (max_Q_next * self._is_terminal)

            # Rebuild RF on newly bootstrapped estimate of Q_next(s, a)
            self._estimator.fit(self._state_action, Q_target)

            ## Metrics
            # Remark: original notebook erroneously used max_Q_next for Q_prev which was recomputed in the loop (= r + 0.1*max_Q_next)
            td_error = np.mean(np.absolute(Q_target - Q_prev))
            meanQ = np.mean(max_Q_next)
            varQ = np.var(max_Q_next)

            td_errors.append(td_error)
            meanQs.append(meanQ)
            varQs.append(varQ)

            # Print metrics every so often averaged over last 10 iterations
            if i % 10 == 0 or i == self._max_iters - 1:
                avg_td_error = np.mean(td_errors[-10:])
                avg_meanQ = np.mean(meanQs[-10:])
                avg_varQ = np.mean(varQs[-10:])
                print('Iter %s/%s: TD-error = %.3f, Q_mean = %.3f, Q_var = %.3f' %
                      (i + 1, self._max_iters, avg_td_error, avg_meanQ, avg_varQ))


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
    fqi = FittedQIteration(TRAINING_DATA, STATE_COLS, 'action', 'reward', 'icustay_id',
                           method='rf', params={'n_estimators': 200, 'max_depth': 5})