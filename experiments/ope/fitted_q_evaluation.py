import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize
from estimators import *


class FittedQEvaluation:
    """ Implementation of the Fitted Q-Evaluation (FQE) estimator for Off-policy
        Policy Evaluation (OPE). For details, see https://arxiv.org/pdf/2102.03607.pdf
    """
    def __init__(self, states, actions, rewards, episodes, method='rf', gamma=0.9, max_iters=25, verbose=False):
        # Terminal state + start state masks
        self._is_start_state = np.insert(episodes[1:] != episodes[:-1], 0, True)  # Checked
        self._is_terminal_state = np.append(episodes[1:] != episodes[:-1], True)  # Checked
        self._action_space = sorted(set(actions[~np.isnan(actions)]))             # Checked

        # Create dataset of ((state, action), reward) pairs
        self._states_actions = self._create_state_action_table(states, actions, non_terminal_only=True)
        self._rewards = rewards[~self._is_terminal_state]

        # Q-table with every possible state-action pair
        self._Q_table = self._create_Q_table(states, self._action_space, next_states_only=True)

        # Estimation method
        if method == 'lasso':
            self._estimator = LassoRegression()
        elif method == 'rf':
            self._estimator = RandomForest()
        else:
            raise Exception('Method %s not understood' % method)

        # Hyper-parameters
        self._gamma = gamma
        self._max_iters = max_iters
        self._verbose = verbose

    def _create_state_action_table(self, states, actions, non_terminal_only=True):
        # Drop terminal states (no action, nor reward)
        if non_terminal_only:
            states = states[~self._is_terminal_state]
            actions = actions[~self._is_terminal_state]

        # Concatenate states and dummy-encoded actions
        num_actions = np.max(self._action_space) + 1
        action_dummies = label_binarize(actions, classes=np.arange(num_actions))
        states_actions = np.column_stack([states, action_dummies])

        return states_actions

    def _create_Q_table(self, states, action_space, next_states_only=True):
        # Drop 'first states' as they are not 'next states'
        if next_states_only:
            states = states[~self._is_start_state]

        # Create table of possible actions
        num_actions = np.max(action_space) + 1
        action_dummies = np.eye(num_actions)

        # Create Q table of |S|x|A| rows with all next-state-next-action combinations
        num_states, state_dims = states.shape
        table = np.column_stack([np.repeat(states, repeats=num_actions, axis=0),          # Next states
                                 np.tile(action_dummies, reps=(num_states, 1)),           # Next actions
                                 np.zeros((num_states * num_actions, 1)),                 # Q-value (initially zero)
                                 np.repeat(np.arange(num_states), repeats=num_actions)])  # Next state ID

        # Create DataFrame for efficiency
        header = ['x%s' % i for i in range(state_dims + num_actions)] + ['policy_q', 'next_state_id']
        return pd.DataFrame(data=table, columns=header)

    def fit(self, policy):
        # Initial fit of estimator setting Q(s, a) = 0
        self._estimator.fit(self._states_actions, np.zeros(self._rewards.shape))

        # Limit pi(a|s) to 'next' actions and 'next' states
        policy = policy[~self._is_start_state].flatten()

        # Separate next-state-next-action part from Q-table
        next_states_actions = self._Q_table.values[:, :-2]

        # Mask states in self._states_actions that are pre-terminal states,
        # i.e. states that transition to terminal states (must not receive V(s'))
        to_terminal = np.roll(self._is_terminal_state, shift=-1)[~self._is_terminal_state]  # Checked

        for it in range(self._max_iters):
            # Predict Q(s', a') using estimator for Q
            q_values = self._estimator.predict(next_states_actions)

            if self._verbose:
                print('It %s/%s: Average Q-value = %.3f' % (it + 1, self._max_iters, np.mean(q_values)))

            # Compute expected SARSA 'next state'-value: sum_a[pi(a|s) * Q(s', a')]
            self._Q_table['policy_q'] = policy * q_values
            next_state_value = self._Q_table.groupby('next_state_id', sort=False)['policy_q'].sum()

            # Bootstrap Q-estimate
            y = self._rewards + self._gamma * (next_state_value * to_terminal)

            # Refit estimator
            self._estimator.fit(self._states_actions, y)

    def Q(self, states, actions):
        # Create state-action table with dummy-encoded actions
        states_actions = self._create_state_action_table(states, actions, non_terminal_only=False)
        return self._estimator.predict(states_actions)

    def V(self, states, policy):
        # Create Q-table with all possible state-action pairs
        Q_table = self._create_Q_table(states, self._action_space, next_states_only=False)

        # Predict Q(s', a') using estimator (limit to state-action part of Q-table)
        states_actions = Q_table.values[:, :-2]
        q_values = self._estimator.predict(states_actions)

        # Predict V(s) = sum_a[pi(a|s) * Q(s, a)]
        Q_table['policy_q'] = policy.flatten() * q_values
        return Q_table.groupby('next_state_id', sort=False)['policy_q'].sum().values


if __name__ == '__main__':
    training_data = pd.read_csv('../datasets/mimic-iii/roggeveen/mimic-iii_train.csv')

    # Unpack training dataset into states, actions, rewards and episode IDs
    no_state_cols = ['icustay_id', 'timestep', 'max_vp_shifted', 'total_iv_fluid_shifted', 'reward', 'action', 'state_sirs']
    actions = training_data['action'].values.astype(np.uint8)
    rewards = training_data['reward'].values
    episodes = training_data['icustay_id'].values.astype(np.uint64)
    states = training_data[[c for c in training_data.columns if c not in no_state_cols]].values

    # Identify start states s0 and action space
    is_start_state = np.insert(episodes[1:] != episodes[:-1], 0, True)

    # Unpack behavior policy
    behavior_df = pd.read_csv('../physician_policy/mimic-iii_train_behavior_policy.csv')
    behavior_policy = behavior_df[[str(i) for i in range(25)]].values

    # Fit FQE
    FQE = FittedQEvaluation(states, actions, rewards, episodes, method='rf', verbose=True)
    FQE.fit(behavior_policy)

    # Predict V(s) an Q(s, a)
    print('Mean Q(s0, a0):', np.mean(FQE.Q(states[is_start_state], actions[is_start_state])))
    print('Mean V(s0):    ', np.mean(FQE.V(states[is_start_state], behavior_policy[is_start_state])))

    # Sanity check: Expected discounted reward?
    rewards = rewards.reshape(-1, 18)[:, :-1]
    gamma = np.power(0.9, np.arange(17))[np.newaxis]
    discounted_reward = np.mean(np.sum(gamma * rewards, axis=1))
    print('\nTrue reward of pi_b:', discounted_reward)
