import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

from importance_sampling import WeightedIS
from fitted_q import FittedQEvaluation, FittedQIteration


class WeightedDoublyRobust:
    def __init__(self, behavior_policy_file, mdp_training_file, mdp_validation_file=None, gamma=0.9, lrate=1e-2,
                 initial_iters=500, warm_iters=100, method='fqe'):
        """ Implementation of the Weighted Doubly Robust (WDR) estimator for OPE. We use a
            Weighted Importance Sampling estimator for the IS part and we use a Fitted Q-
            Evaluation (FQE) or a Fitted Q-Iteration (FQI) estimator for the DM part.
            Please see https://arxiv.org/pdf/1802.03493.pdf for more details.

            Params
            behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                  behavior policy, chosen actions ('action') and associated rewards ('reward').
            mdp_training_file:    Path to DataFrame containing states, actions and rewards of training set.
            mdp_validation_file:  Path to DataFrame containing states, actions and rewards of validation set.
            gamma:                Discount factor
        """
        # Define WIS estimator
        self._weighted_is = WeightedIS(behavior_policy_file, gamma=gamma)
        self.timesteps = self._weighted_is.timesteps
        self.actions = self._weighted_is.actions
        self.gamma = gamma

        # Define covariate estimation method (FQI/FQE)
        self._method = method
        if self._method == 'fqi':
            estimator = FittedQIteration
        elif self._method == 'fqe':
            estimator = FittedQEvaluation
        else:
            raise Exception('Estimator %s not recognized' % estimation_type)

        # If no validation set is given, assume training set
        mdp_validation_file = mdp_validation_file if mdp_validation_file is not None else mdp_training_file

        # Init DM estimator
        self._fitted_estimator = estimator(mdp_training_file, mdp_validation_file, gamma=gamma, lrate=lrate,
                                           initial_iters=initial_iters, warm_iters=warm_iters)
        self._fitted = False

    def _to_table(self, arr):
        return np.nan_to_num(arr.reshape(-1, self.timesteps), nan=0.0)  # Replace NaNs in reward table

    def fit(self, train_pi_e):
        """ (Re)fits FQE/FQI estimator on the training set given action probabilities
            over training states. Note, if FQI is used, `train_pi_e` will not be used.
        """
        if self._method == 'fqi' and not self._fitted:
            self._fitted_estimator.fit()  # Remark: Only needs to be fitted once as independent form policy!
        elif self._method == 'fqe':
            self._fitted_estimator.fit(train_pi_e)
        self._fitted = True
        return self

    def __call__(self, pi_e):
        """ Computes the WIS estimate of V^πe.

            evaluation_policy: Table of action probs acc. to πe with shape (num_states, num_actions)

            Returns:           Estimate of mean V^πe
        """
        # Compute importance weights and array of discount factors
        weights = self._weighted_is(pi_e, return_weights=True)
        gamma = np.power(self.gamma, np.arange(self.timesteps))[np.newaxis]

        # TODO: Move weights from t-1 to t
        next_weights = weights

        # Table of Q estimates (limited to actions chosen by physician)
        q = self._fitted_estimator.state_action_value()
        q = self._to_table(np.take_along_axis(q, self.actions, axis=1))

        # Table of V estimates of visited states
        v = self._to_table(self._fitted_estimator.state_value(pi_e))

        # Computes WDR estimate
        return self._weighted_is(pi_e) - np.sum(gamma * (next_weights * q - weights * v))


if __name__ == '__main__':
    # Behavior policy
    behavior_policy_path = 'physician_policy/roggeveen_4h/mimic-iii_%s_behavior_policy.csv'
    train_behavior_policy = pd.read_csv(behavior_policy_path % 'train').filter(regex='\d+').values
    valid_behavior_policy = pd.read_csv(behavior_policy_path % 'valid').filter(regex='\d+').values

    # Random policy
    train_random_policy = np.random.uniform(0, 1, train_behavior_policy.shape)
    train_random_policy = train_random_policy / np.sum(train_random_policy, axis=1, keepdims=True)

    valid_random_policy = np.random.uniform(0, 1, valid_behavior_policy.shape)
    valid_random_policy = valid_random_policy / np.sum(valid_random_policy, axis=1, keepdims=True)

    # Fit estimators on policies evaluated on training data
    behavior_policy_file = 'physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv'
    mdp_training_file = '../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv'
    mdp_validation_file = '../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv'
    wdr_behavior = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, mdp_validation_file).fit(train_behavior_policy)
    wdr_random = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, mdp_validation_file).fit(train_random_policy)

    print('WDR:')
    print('behavior:', wdr_behavior(valid_behavior_policy))
    print('random:  ', wdr_random(valid_random_policy))