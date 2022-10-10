import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import label_binarize


class FQE:
    def __init__(self, training_file, state_cols, action_col, reward_col, episode_col, method='rf', gamma=0.9,
                 reg_lambda=0.001, max_iters=250, num_estimators=800, max_depth=5):
        """ Implementation of the Fitted Q-Evaluation (FQE) method for OPE.
            Please refer to https://arxiv.org/pdf/2107.11003.pdf for details.

            Params
            training_file:        DataFrame containing state features and integer actions used to train FQE model
            state_cols:           List of columns corresponding to state-space features
            action_col:           Column corresponding to action chosen
            reward_col:           Column corresponding to reward received
            episode_col:          Column with episode identifiers

            method:               Model used to approximate Q-function ('rf'|'lr', default: 'rf')
            gamma:                Discount factor
            reg_lambda:           Regularization factor used to suppress unseen actions (when method='lr')
            max_iters:            Maximum number of iterations to run
            num_estimators:       Number of trees to use (when method='rf')
            max_depth:            Maximum depth trees may grow  (when method='rf')
        """
        # Create dataset of state-action-tables, next-state-next-action tables and rewards
        df = pd.read_csv(training_file)
        self._state_action, self._reward = self._create_state_action_table(df, state_cols, action_col, reward_col)
        self._next_state_action, self._is_next_state = self._create_next_state_action_table(df, state_cols, action_col, episode_col)

        # Estimators
        if method == 'rf':
            self._estimator = RandomForestRegressor(num_estimators, max_depth=max_depth, random_state=1)
        elif method == 'lr':
            self._estimator = Lasso(reg_lambda, random_state=1)
        else:
            raise Exception('Method %s not understood' % method)

        # Hyper parameters
        self._gamma = gamma
        self._max_iters = max_iters

    @staticmethod
    def _create_state_action_table(df, state_cols, action_col, reward_col):
        """ Creates table of shape SxA consisting of states `s` and one-hot
            encoded actions `a`. and a reward vector `r`.
        """
        # Unpack training set into states, actions and rewards
        non_terminals = df['reward'].notna()
        states = df[non_terminals][state_cols].values
        actions = df[non_terminals][action_col].values.astype(int)
        rewards = df[non_terminals][reward_col].values

        num_actions = int(np.max(df[action_col]) + 1)

        # Concatenate states and actions (using one-hot)
        actions_one_hot = label_binarize(actions, classes=range(num_actions))
        state_action_table = np.column_stack([states, actions_one_hot])
        return state_action_table, rewards

    @staticmethod
    def _create_next_state_action_table(df, state_cols, action_col, episode_col):
        """ Creates table of shape SxA with rows consisting of states `s'`,
            one-hot encoded actions `a'`, Q-target estimates and next-state ids.
        """
        # Drop 'first states' as they are not 'next states'
        is_next_state = df.groupby(episode_col)[episode_col].transform(lambda ep: np.arange(ep.shape[0]) > 0).values
        df = df[is_next_state]

        # Count follow-up states
        next_states = df[state_cols].values
        num_states = next_states.shape[0]

        # Count available actions
        num_actions = int(np.max(df[action_col]) + 1)  # -> 25 actions
        next_actions = np.eye(num_actions)

        # Create table with all next-state-next-action combinations
        next_state_ids = np.arange(num_states)
        table = np.column_stack([np.repeat(next_states, repeats=num_actions, axis=0),      # Next states
                                 np.tile(next_actions, reps=(num_states, 1)),              # Next actions
                                 np.repeat(next_state_ids, repeats=num_actions, axis=0)])  # next state ids (for bookkeeping)

        header = state_cols + list(range(num_actions)) + ['next_state_id']
        table = pd.DataFrame(data=table, columns=header)

        return table, is_next_state

    def _fit_estimator(self, pi):
        """ Iteratively estimates V^πe using Fitted Q-Evaluation

            pi:  Table of action probs acc. to policy π with shape (num_states, num_actions)
        """
        # Initial fit of estimator setting Q(s, a) = 0
        self._estimator.fit(self._state_action, np.zeros(self._reward.shape))

        # Keep only follow-up states πe(a'|s')
        pi_next = pi[self._is_next_state].flatten()

        # Create next_state-next_action table
        state_action_cols = ~self._next_state_action.columns.isin(['next_state_id', 'V_pi'])
        next_state_action = self._next_state_action.loc[:, state_action_cols].values

        # Iterative bootstrap estimate
        for i in range(self._max_iters):
            # Compute πe(a'|s') * Q(a'|s')
            self._next_state_action['V_pi'] = pi_next * self._estimator.predict(next_state_action)

            # Compute expected state value; yi = reward + gamma * sum[πe(a|s) * Q(s, a)]
            V_next = self._next_state_action.groupby('next_state_id', sort=False)['V_pi'].sum().values
            yi = self._reward + self._gamma * V_next

            # Refit Q-estimator
            self._estimator.fit(self._state_action, yi)

            print(self._estimator.score(self._state_action, yi))

    def __call__(self, pi):
        """ Computes the FQE estimate of V^πe.

            pi:       Table of action probs acc. to to-be-evaluated policy π with shape (num_states, num_actions)
            Returns:  FQE estimate of mean V^πe
        """
        # Fit estimator for Q(s, a)
        self._fit_estimator(pi)

        # Estimate sum_a[πe(a|s) * Q(s, a)] / N
        return self._next_state_action.groupby('next_state_id', sort=False)['V_pi'].sum().mean()


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

    # Behavior policy
    behavior_df = pd.read_csv('physician_policy/mimic-iii_train_behavior_policy.csv')
    behavior_policy = behavior_df[[str(i) for i in range(25)]].values

    # Random policy
    random_policy = np.random.uniform(0, 1, behavior_policy.shape)
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    print('FQE:')
    print('behavior:', fqe(behavior_policy))
    print('random:  ', fqe(random_policy))

    # Sanity check: estimate true discounted reward
    rewards = behavior_df['reward'].values.reshape(-1, 18)[:, :-1]
    gamma = np.power(0.9, np.arange(17))[np.newaxis]
    discounted_reward = np.mean(np.sum(gamma * rewards, axis=1))
    print('\nTrue reward of pi_b:', discounted_reward)
