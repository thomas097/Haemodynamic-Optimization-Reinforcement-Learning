import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso


class MLP(torch.nn.Module):
    """ Simple MLP to approximate Q-function """
    def __init__(self, hidden_units=32, lrate=1e-3):
        super().__init__()
        self._layer1 = None
        self._layer2 = None
        self._optimizer = None
        self._hidden_units = hidden_units
        self._lrate = lrate
        self._leaky_relu = torch.nn.LeakyReLU()
        self._criterion = torch.nn.MSELoss()

    def forward(self, X):
        h = self._leaky_relu(self._layer1(X))
        return self._layer2(h)

    def predict(self, X):
        return self(torch.Tensor(X)).detach().numpy()[:, 0]

    def fit(self, X, y):
        X, y = torch.Tensor(X), torch.Tensor(y)[:, None]

        if self._layer1 is None:
            self._layer1 = torch.nn.Linear(X.shape[1], self._hidden_units)
            self._layer2 = torch.nn.Linear(self._hidden_units, 1)
            self._optimizer = torch.optim.Adam(self.parameters(), lr=self._lrate)

        self._optimizer.zero_grad()
        loss = self._criterion(self(X), y)
        loss.backward()
        self._optimizer.step()

        return loss.item()


class FittedQEvaluation:
    def __init__(self, training_data, state_cols, action_col, reward_col, episode_col, method='rf', params={},
                 max_iters=100, gamma=0.9):
        """ Implementation of Fitted Q-Evaluation (FQE) for OPE.
            Please refer to https://arxiv.org/pdf/2107.11003.pdf for details.

            training_file:        DataFrame containing state features and integer actions used to train FQE model
            state_cols:           List of columns corresponding to state-space features
            action_col:           Column corresponding to action chosen
            reward_col:           Column corresponding to reward received
            episode_col:          Column with episode identifiers

            method:  Estimator method to use ('rf'|'lr'|'mlp', default: 'rf')
            params:  Hyperparameters of method (see Sklearn docs)
            gamma:   Discount factor
        """
        # Import dataset
        df = pd.read_csv(training_data, usecols=state_cols + [action_col, reward_col, episode_col]).reset_index(drop=True)

        # Get non-terminal states
        self._states = df[df[reward_col].notna()][state_cols].values

        # Pair rewards and actions with next state (needed for Q-update)
        self._num_episodes = len(df[episode_col].unique())
        df[action_col] = df.groupby(episode_col)[action_col].shift(1)
        df[reward_col] = df.groupby(episode_col)[reward_col].shift(1)

        # Drop 'start states' as they aren't 'next states' (with actions and rewards)
        df_next = df[df[action_col].notna()]

        # Extract next states, actions and rewards
        self._is_next_states = df_next.index.values
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
        elif method == 'mlp':
            self._estimator = MLP(hidden_units=params['hidden_units'])
        else:
            raise Exception('Method %s not understood' % method)

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

    def _fit(self, pi):
        """ Iteratively fits Q-estimator to data. """
        # Record metrics over time
        td_errors, meanQs, varQs = [], [], []

        # Initial fit setting Q(s, a) = r
        self._estimator.fit(self._state_action, self._rewards)

        # Evaluate policy at next states
        pi_next = pi[self._is_next_states].flatten()

        # Drop identifier from next-state-next-action table
        next_state_action = self._next_state_action.values[:, :-1]

        # Perform Q-iteration
        for i in range(self._max_iters):
            # Initial guess of Q(s, a) by the estimator
            Q_prev = self._estimator.predict(self._state_action)

            # Predict Q(s', a') for every state s' and action a'
            self._next_state_action['Q_next'] = pi_next * self._estimator.predict(next_state_action)

            # Recompute Q(s, a) = r + gamma * sum_a'[pi(a'|s') * Q(s', a')]
            weighted_Q_next = self._next_state_action.groupby('next_state', sort=False)['Q_next'].sum()
            Q_target = self._rewards + self._gamma * (weighted_Q_next * self._is_terminal)

            # Rebuild RF on newly bootstrapped estimate of Q_next(s, a)
            self._estimator.fit(self._state_action, Q_target)

            ## Metrics
            # Remark: original notebook erroneously used max_Q_next for Q_prev which was recomputed in the loop (= r + 0.1*max_Q_next)
            td_error = np.mean(np.absolute(Q_target - Q_prev))
            meanQ = np.mean(weighted_Q_next)
            varQ = np.var(weighted_Q_next)

            td_errors.append(td_error)
            meanQs.append(meanQ)
            varQs.append(varQ)

            # Print metrics every so often averaged over last 10 iterations
            if i % 10 == 0 or i == self._max_iters - 1:
                avg_td_error = float(np.mean(td_errors[-10:]))
                avg_meanQ = float(np.mean(meanQs[-10:]))
                avg_varQ = float(np.mean(varQs[-10:]))
                print('Iter %s/%s: TD-error = %.3f, Q_mean = %.3f, Q_var = %.3f' %
                      (i + 1, self._max_iters, avg_td_error, avg_meanQ, avg_varQ))

    def predict(self, state_actions):
        """ Predict V^πe estimate of state-action pairs using estimator """
        return self._estimator.predict(state_actions)

    def __call__(self, pi):
        """ Computes the FQE estimate of V^πe.

            pi:       Table of action probs acc. to to-be-evaluated policy π with shape (num_states, num_actions)
            Returns:  FQE estimate of mean V^πe
        """
        # Fit estimator for Qπe(s, a)
        self._fit(pi)

        # Determine start states
        num_time_steps = (self._state_action.shape[0] // self._num_episodes) - 1
        start_states = self._state_action[0::num_time_steps]

        # Predict mean[V(s0)]
        return self.predict(start_states).mean()


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
    fqi = FittedQEvaluation(TRAINING_DATA, STATE_COLS, 'action', 'reward', 'icustay_id', max_iters=300,
                            method='mlp', params={'lrate': 0.001, 'hidden_units': 128})

    # Behavior policy
    behavior_df = pd.read_csv('physician_policy/mimic-iii_train_behavior_policy.csv')
    behavior_policy = behavior_df[[str(i) for i in range(25)]].values
    print('FQE:\nbehavior:', fqi(behavior_policy))

    # Sanity check: estimate true discounted reward
    rewards = behavior_df['reward'].values.reshape(-1, 18)[:, :-1]
    gamma = np.power(0.9, np.arange(17))[np.newaxis]
    discounted_reward = np.mean(np.sum(gamma * rewards, axis=1))
    print('\nTrue reward of pi_b:', discounted_reward)