import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

from importance_sampling import WeightedIS
from fitted_q_evaluation import FittedQEvaluation


class WeightedDoublyRobust:
    def __init__(self, behavior_policy_file, mdp_training_file, gamma=1.0, lrate=1e-2, iters=1000,
                 method='fqe', reward_bounds=(-15, 15)):
        """ Implementation of the Weighted Doubly Robust (WDR) estimator for OPE. We use
            a Weighted Importance Sampling estimator for the IS part, and we use a Fitted
            Q-Evaluation (FQE) estimator for the DM part.
            Please see https://arxiv.org/pdf/1802.03493.pdf for more details.

            Params
            behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                  behavior policy, chosen actions ('action') and associated rewards ('reward').
            mdp_training_file:    Path to DataFrame containing states, actions and rewards of training set.
            gamma:                Discount factor
        """
        # Define WIS estimator
        self._weighted_is = WeightedIS(behavior_policy_file, gamma=gamma)

        # Parameters
        self.timesteps = self._weighted_is.timesteps
        self.actions = self._weighted_is.actions
        self.gamma = gamma
        self.min_reward, self.max_reward = reward_bounds

        # DR estimator
        if method == 'fqe':
            self._estimator = FittedQEvaluation(mdp_training_file, gamma=gamma, lrate=lrate, iters=iters)
        elif method == 'fqi':
            self._estimator = FittedQIteration(mdp_training_file, gamma=gamma, lrate=lrate, iters=iters)
        else:
            raise Exception('method not understood: method = "fqe"|"fqi"')

    def _to_table(self, arr):
        return np.nan_to_num(arr.reshape(-1, self.timesteps), nan=0.0)  # Replace NaNs in reward table

    def fit(self, train_pi_e):
        """ Fits FQE estimator on the training set given action probabilities
            over training states
        """
        self._estimator.fit(train_pi_e)
        return self

    def __call__(self, pi_e):
        """ Computes the WIS estimate of V^πe.

            evaluation_policy: Table of action probs acc. to πe with shape (num_states, num_actions)

            Returns:           Estimate of mean V^πe
        """
        # Compute importance weights and array of discount factors
        weights = self._weighted_is(pi_e, return_weights=True)
        gamma = np.power(self.gamma, np.arange(self.timesteps))[np.newaxis]

        # TODO: Move weights from t-1 to t (what to do for t=-1?)
        next_weights = weights

        # Table of Q estimates (limited to actions chosen by physician)
        q = self._estimator.state_action_value()
        q = self._to_table(np.take_along_axis(q, self.actions, axis=1))

        # Table of V estimates of visited states
        v = self._to_table(self._estimator.state_value(pi_e))

        # Clip Q and V to min/max reward
        q = np.clip(q, a_min=self.min_reward, a_max=self.max_reward)
        v = np.clip(v, a_min=self.min_reward, a_max=self.max_reward)

        # Computes WDR estimate
        return self._weighted_is(pi_e) - np.sum(gamma * (next_weights * q - weights * v))


if __name__ == '__main__':
    # Behavior policy
    behavior_policy_path = 'physician_policy/roggeveen_4h/mimic-iii_%s_behavior_policy.csv'
    train_behavior_policy = pd.read_csv(behavior_policy_path % 'train').filter(regex='\d+').values

    # Random policy
    train_random_policy = np.random.uniform(0, 1, train_behavior_policy.shape)
    train_random_policy = train_random_policy / np.sum(train_random_policy, axis=1, keepdims=True)

    # Needed data files
    behavior_policy_file = 'physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv'
    mdp_training_file = '../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv'

    # Fit FQE-WDR estimators
    wdr_behavior = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method='fqe').fit(train_behavior_policy)
    wdr_random = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method='fqe').fit(train_random_policy)

    print('WDR-FQE:')
    print('behavior:', wdr_behavior(train_behavior_policy))
    print('random:  ', wdr_random(train_random_policy))

    # Fit FQI-WDR estimators
    wdr_behavior = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method='fqi').fit(train_behavior_policy)
    wdr_random = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method='fqi').fit(train_random_policy)

    print('WDR-FQI:')
    print('behavior:', wdr_behavior(train_behavior_policy))
    print('random:  ', wdr_random(train_random_policy))