import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize

from per_horizon_importance_sampling import PHWIS
from fitted_q_evaluation import FittedQEvaluation
from fitted_q_iteration import FittedQIteration


class WeightedDoublyRobust:
    def __init__(self, behavior_policy_file, mdp_training_file, lrate=1e-2, iters=1000, method='fqe',
                 reward_bounds=(-100, 100)):
        """ Implementation of the Weighted Doubly Robust (WDR) estimator for Off-policy Policy Evaluation (OPE).
        We use a Per-Horizon Weighted Importance Sampling (PHWIS) estimator for the IS part and a Fitted Q-Evaluation
        (FQE) or Iteration (FQI) estimator for the DM part. For details, see https://arxiv.org/pdf/1911.06854.pdf
        :param behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                     behavior policy, chosen actions ('action') and associated rewards ('reward')
        :param mdp_training_file:    Path to DataFrame containing states, actions and rewards of training set
        """
        self._min_reward, self._max_reward = reward_bounds
        self._episodes = pd.read_csv(behavior_policy_file).episode

        # importance sampling (IS) estimator
        self.phwis = PHWIS(behavior_policy_file)

        # direct method (DM) estimator (FQE or FQI)
        estimator = FittedQEvaluation if method == 'fqe' else FittedQIteration
        self.dm = estimator(
            training_file=mdp_training_file,
            gamma=1.0, # per-horizon estimators do not know discounting
            lrate=lrate,
            iters=iters,
            reward_range=reward_bounds
        )

    def fit(self, train_pi_e):
        """ Fits FQE/FQI estimator on the training set given action probabilities
        over training states by the evaluation policy
        :param train_pi_e:  Tensor of action probabilities at each training set state
        :returns:           Reference to self
        """
        self.dm.fit(train_pi_e)
        return self

    def __call__(self, pi_e, episodes=None):
        """ Computes the WDR estimate of V^πe
        :param pi_e:      Table of action probs acc. to πe with shape (num_states, num_actions)
        :param episodes:  Subset of episodes of pi_e to consider in estimating V^πe
        :returns:         Estimate of mean V^πe
        """
        # wis and importance weights
        wis = self.phwis(pi_e, episodes=episodes)
        ratios = self.phwis.ratios(pi_e, episodes=episodes)

        # compute weight at t-1 by shifting the computed ratios forward by one timestep
        # Note : as rho_t-1 is not known at t=0, we choose to drop the first time step of
        # each admission by setting rho_t = rho_t-1 = 0
        ratios['rho_prev'] = ratios.groupby('episode').rho.shift(periods=1)
        ratios.loc[ratios.rho_prev.isna(), ['rho', 'rho_prev']] = 0
        rho = ratios.rho * ratios.wh
        rho_prev = ratios.rho_prev * ratios.wh

        # estimate using model V and Q covariates
        v = self.dm.state_value(pi_e, episodes=episodes)
        q = self.dm.state_action_value(episodes=episodes)

        # limit V- and Q-values to selected episodes and actions chosen by behavior policy
        actions = self.phwis._actions[self._episodes.isin(episodes)]
        q = np.take_along_axis(q, actions, axis=1).flatten()

        # clip expected rewards to min/max reward
        v = np.clip(v, a_min=self._min_reward, a_max=self._max_reward)
        q = np.clip(q, a_min=self._min_reward, a_max=self._max_reward)

        # WDR estimate
        return wis - np.sum((rho * q) - (rho_prev * v))



if __name__ == '__main__':
    behavior_policy_file = 'physician_policy/amsterdam-umc-db_aggregated_full_cohort_1h_knn/valid_behavior_policy.csv'
    mdp_training_file = '../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h/valid.csv'

    # Toy policies to evaluate
    behavior_policy = pd.read_csv(behavior_policy_file).filter(regex='\d+').values

    random_policy = np.random.uniform(0, 1, behavior_policy.shape)
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    # Fit FQE-WDR estimators
    print('WDR (FQE)')
    wdr_behavior = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method='fqe').fit(behavior_policy)
    print('behavior:  ', wdr_behavior(behavior_policy))

    wdr_random = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method='fqe').fit(random_policy)
    print('random:    ', wdr_random(random_policy))