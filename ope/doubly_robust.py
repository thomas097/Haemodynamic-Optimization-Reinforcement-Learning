import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

from importance_sampling import WIS
from fitted_q_evaluation import FittedQEvaluation


class WeightedDoublyRobust:
    def __init__(self, behavior_policy_file, mdp_training_file, gamma=1.0, lrate=1e-2, iters=2000, method='fqe',
                 reward_bounds=(-15, 15)):
        """ Implementation of the Weighted Doubly Robust (WDR) estimator for OPE. We use a Weighted
        Importance Sampling estimator for the IS part, and we use a Fitted Q-Evaluation (FQE) estimator
        for the DM part. Please see https://arxiv.org/pdf/1802.03493.pdf for more details
        :param behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                     behavior policy, chosen actions ('action') and associated rewards ('reward')
        :param mdp_training_file:    Path to DataFrame containing states, actions and rewards of training set
        :param gamma:                Discount factor
        """
        # importance sampling (IS) estimator
        self._wis = WIS(behavior_policy_file, gamma=gamma)

        # parameters
        self._n_timesteps = self._wis._n_timesteps
        self._n_episodes = self._wis._n_episodes
        self._actions = self._wis._actions
        self._gammas = np.power(gamma, np.arange(self._n_timesteps))[np.newaxis]
        self._min_reward, self._max_reward = reward_bounds

        # direct method (DM) estimator
        if method == 'fqe':
            self._estimator = FittedQEvaluation(mdp_training_file, gamma=gamma, lrate=lrate, iters=iters)
        elif method == 'fqi':
            self._estimator = FittedQIteration(mdp_training_file, gamma=gamma, lrate=lrate, iters=iters)
        else:
            raise Exception('method not understood: method = "fqe"|"fqi"')

    def fit(self, train_pi_e):
        """ Fits FQE/FQI estimator on the training set given action probabilities
        over training states by the evaluation policy
        :param train_pi_e:  Tensor of action probabilities at each training set state
        :returns:           Reference to self
        """
        self._estimator.fit(train_pi_e)
        return self

    def __call__(self, pi_e):
        """ Computes the WDR estimate of V^πe.
        :param evaluation_policy: Table of action probs acc. to πe with shape (num_states, num_actions)
        :returns:                 Estimate of mean V^πe
        """
        # importance weights (byproduct of WIS)
        wis = self._wis(pi_e)
        rho = self._wis.ratios / np.sum(self._wis.ratios, axis=0, keepdims=True)

        # note: we adopt the COBS fix for t-1, i.e. rho = 1 / n_episodes,
        # see: https://arxiv.org/pdf/1911.06854.pdf
        rho_t_minus_one = np.ones((self._n_episodes, 1)) / self._n_episodes
        rho_prev = np.column_stack([rho_t_minus_one, rho[:, :-1]])

        # compute control covariates (limit q to actions by physician)
        v = self._estimator.state_value(pi_e).reshape(-1, self._n_timesteps)
        q = self._estimator.state_action_value()
        q = np.take_along_axis(q, self._actions, axis=1).reshape(-1, self._n_timesteps)

        # clip expected rewards to min/max reward
        q = np.clip(q, a_min=self._min_reward, a_max=self._max_reward)
        v = np.clip(v, a_min=self._min_reward, a_max=self._max_reward)

        # WDR estimate
        return wis - np.sum(self._gammas * (rho * q - rho_prev * v))




if __name__ == '__main__':
    behavior_policy_file = 'physician_policy/aggregated_4h/mimic-iii_valid_behavior_policy.csv'
    mdp_training_file = '../preprocessing/datasets/mimic-iii/aggregated_4h/mimic-iii_valid.csv'

    # Policies
    behavior_policy = pd.read_csv(behavior_policy_file).filter(regex='\d+').values

    random_policy = np.random.uniform(0, 1, behavior_policy.shape)
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    zerodrug_policy = np.full(shape=behavior_policy.shape, fill_value=1e-6)
    zerodrug_policy[:, 0] = 1 - (1e-6 * behavior_policy.shape[1])

    # Fit FQE-WDR estimators
    wdr_behavior = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method='fqe').fit(behavior_policy)
    wdr_random = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method='fqe').fit(random_policy)
    wdr_zerodrug = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method='fqe').fit(zerodrug_policy)

    print('WDR-FQE:')
    print('behavior: ', wdr_behavior(behavior_policy))
    print('random:   ', wdr_random(random_policy))
    print('zero drug:', wdr_zerodrug(zerodrug_policy))

    # # Fit FQI-WDR estimators
    # wdr_behavior = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method='fqi').fit(train_behavior_policy)
    # wdr_random = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method='fqi').fit(train_random_policy)
    #
    # print('WDR-FQI:')
    # print('behavior:', wdr_behavior(train_behavior_policy))
    # print('random:  ', wdr_random(train_random_policy))