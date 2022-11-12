import numpy as np
import pandas as pd


class IS:
    def __init__(self, behavior_policy_file, gamma=1.0, bootstraps=0, conf=0.95):
        """ Implementation of Stepwise Importance Sampling (IS) for Off-policy Policy Evaluation (OPE)
        Optionally the method uses a non-parametric bootstrap to obtain a lower/upper confidence bound
        Please refer to https://arxiv.org/pdf/1807.01066.pdf for details.

        :param behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                     behavior policy, chosen actions ('action') and associated rewards ('reward').
        :param gamma:                Discount factor (default: 1.0)
        :param bootstraps:           How many bootstrap sets to use to estimate confidence bounds (default: 0; disabled)
        :param conf:                 Confidence level (CL) of the bootstrap estimate (default: CL 95%)
        """
        df = pd.read_csv(behavior_policy_file)

        # test whether all episodes are of the same length
        timesteps = df.groupby('episode').size().values
        if np.any(timesteps != timesteps[0]):
            raise exception('Episodes must be of the same length. Found %s' % np.unique(timesteps))

        # behavior policy's actions and action probabilities
        self.actions = np.expand_dims(df['action'], axis=1).astype(int)
        self.pi_b = self._behavior_policy(df)

        self.timesteps = timesteps[0]
        self.rewards = self._to_table(df['reward'].values)
        self.gammas = np.power(gamma, np.arange(self.timesteps))[np.newaxis]

        # bootstrapping
        self._bootstraps = bootstraps
        self._conf = conf

    @staticmethod
    def _behavior_policy(df):
        # Return numeric columns assumed to belong to actions, i.e., '0'-'24'
        return df.filter(regex='\d+').values.astype(np.float64)

    def _to_table(self, arr):
        return arr.reshape(-1, self.timesteps)

    @staticmethod
    def _estimate(gammas, weights, rewards):
        """ Computes IS estimate given weights and discounted rewards
        :param gammas:  ndarray of discount factors of shape (episodes, timesteps)
        :param weights: ndarray of importance weights of shape (episodes, timesteps)
        :param rewards: ndarray of rewards of shape (episodes, timesteps)
        :returns:       IS estimate of V^πe
        """
        return np.mean(np.sum(gammas * weights * rewards, axis=1))

    def __call__(self, pi_e):
        """ Computes the IS estimate of V^πe
        :param pi_e:  Table of action probs acc. to πe with shape (num_states, num_actions)
        :returns:     Estimate of mean V^πe. If bootstraps > 0, a tuple (lower, median, upper) is returned
        """
        # Extract table of action probabilities acc. to πe (πb) w.r.t. chosen actions
        pi_b = self.pi_b
        action_probs_e = self._to_table(np.take_along_axis(pi_e, self.actions, axis=1))
        action_probs_b = self._to_table(np.take_along_axis(pi_b, self.actions, axis=1))

        # Compute cumulative importance ratio
        weights = np.cumprod(action_probs_e / action_probs_b, axis=1)

        # estimate distribution of scores by bootstrapping
        if self._bootstraps > 0:
            estimates = []
            for _ in range(self._bootstraps):
                # sample with replacement a bootstrap set of size |episodes|
                i = np.random.choice(np.arange(weights.shape[0]), size=weights.shape[0], replace=True)
                estimates.append(self._estimate(self.gammas, weights[i], self.rewards[i]))

            median = np.median(estimates)
            lower = np.quantile(estimates, q=(1 - self._conf) / 2)
            upper = np.quantile(estimates, q=(1 + self._conf) / 2)
            return lower, median, upper

        # returns estimate over the complete test set
        return self._estimate(self.gammas, weights, self.rewards)


class WeightedIS(IS):
    def __init__(self, behavior_policy_file, gamma=1.0, bootstraps=0, conf=0.95, verbose=False):
        """ Implementation of Stepwise Weighted Importance Sampling (WIS) for Off-policy Policy Evaluation (OPE)
        Optionally the method uses a non-parametric bootstrap to obtain a lower/upper confidence bound
        Please refer to https://arxiv.org/pdf/1807.01066.pdf for details
        :param behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                  behavior policy, chosen actions ('action') and associated rewards ('reward').
        :param gamma:                Discount factor (default: 1.0)
        :param verbose:              Whether to print information such as effective sample size (ESS) to stdout.
        :param bootstraps:           How many bootstrap sets to use to estimate confidence bounds (default: 0; disabled)
        :param conf:                 Confidence level (CL) of the bootstrap estimate (default: CL 95%)
        """
        super().__init__(behavior_policy_file, gamma=gamma, bootstraps=bootstraps, conf=conf)
        self.verbose = verbose

    @staticmethod
    def _effective_sample_size(ratios):
        """ Estimates effective sample size of estimator
        See: http://www.nowozin.net/sebastian/blog/effective-sample-size-in-importance-sampling.html
        :param ratios:  Unnormalized importance ratios
        :return:        Effective sample size
        """
        weights = ratios / ratios.sum(axis=0, keepdims=True)
        return 1 / np.sum(weights[:, -1] ** 2)

    @staticmethod
    def _estimate(gammas, ratios, rewards):
        """ Computes WIS estimate given weights and discounted rewards
        :param gammas:  ndarray of discount factors of shape (episodes, timesteps)
        :param ratios: ndarray of non-normalized importance ratios of shape (episodes, timesteps)
        :param rewards: ndarray of rewards of shape (episodes, timesteps)
        :returns:       IS estimate of V^πe
        """
        weights = ratios / ratios.sum(axis=0, keepdims=True)
        return np.sum(gammas * weights * rewards)

    def __call__(self, pi_e):
        """ Computes the WIS estimate of V^πe
        :param pi_e:  Table of action probs acc. to πe with shape (num_states, num_actions)
        :returns:     Estimate of mean V^πe. If bootstraps > 0, a tuple (lower, median, upper) is returned
        """
        # extract action probabilities acc. to πe (πb) w.r.t. chosen actions
        pi_b = self.pi_b
        action_probs_e = self._to_table(np.take_along_axis(pi_e, self.actions, axis=1))
        action_probs_b = self._to_table(np.take_along_axis(pi_b, self.actions, axis=1))

        # compute cumulative importance ratio
        ratios = np.cumprod(action_probs_e / action_probs_b, axis=1) ** (1/2)
        if self.verbose:
            print('Effective sample size: %.1f' % self._effective_sample_size(ratios))

        # estimate distribution of scores by bootstrapping
        if self._bootstraps > 0:
            estimates = []
            for _ in range(self._bootstraps):
                # sample with replacement a bootstrap set of size |episodes|
                i = np.random.choice(np.arange(ratios.shape[0]), size=ratios.shape[0], replace=True)
                estimates.append(self._estimate(self.gammas, ratios[i], self.rewards[i]))

            median = np.median(estimates)
            lower = np.quantile(estimates, q=(1 - self._conf) / 2)
            upper = np.quantile(estimates, q=(1 + self._conf) / 2)
            return lower, median, upper

        # returns estimate over the complete test set
        return self._estimate(self.gammas, ratios, self.rewards)


if __name__ == '__main__':
    # behavior policy
    behavior_policy_file = 'physician_policy/roggeveen_4h_with_cv/mimic-iii_valid_behavior_policy.csv'
    behavior_df = pd.read_csv(behavior_policy_file)
    behavior_policy = behavior_df.filter(regex='\d+').values  # -> 25 actions marked by integers 0 to 24!

    # dr. House policy (mimics the physician if patient survives (i.e. does the right thing), but goes crazy otherwise)
    from sklearn.preprocessing import label_binarize
    house_policy = label_binarize(behavior_df.action, classes=np.arange(25))
    crazy_action = label_binarize([1], classes=np.arange(25))
    for i, r in enumerate(behavior_df.reward):
        if r == -15:
            house_policy[i-17:i + 1] = crazy_action  # crazy_action is impossible (VP but no IV)

    # random policy
    random_policy = np.random.uniform(0, 1, behavior_policy.shape)  # Add noise
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    # zero-drug policy
    zerodrug_policy = np.full(behavior_policy.shape, fill_value=1e-6)
    zerodrug_policy[:, 0] = 1 - (24 * 1e-6)

    # estimate confidence intervals over WIS scores using 1000 bootstrap sets
    wimp_sampling = WeightedIS(behavior_policy_file, bootstraps=1000, verbose=True)
    conf_intervals = np.array([
        wimp_sampling(behavior_policy),
        wimp_sampling(house_policy),
        wimp_sampling(random_policy),
        wimp_sampling(zerodrug_policy)
    ])

    # Plot confidence bounds as errorplot
    import matplotlib.pyplot as plt

    plt.figure()
    lower_err = conf_intervals[:, 1] - conf_intervals[:, 0]
    upper_err = conf_intervals[:, 2] - conf_intervals[:, 1]
    plt.errorbar(x=[1, 2, 3, 4], y=conf_intervals[:, 1], yerr=[lower_err, upper_err], fmt='o', color='k')
    plt.xticks([1, 2, 3, 4], labels=['physician', 'dr. House', 'random', 'zero-drug'])
    plt.xlabel('policy')
    plt.ylabel('WIS')
    plt.show()

    # Sanity check: estimate true reward
    rewards = behavior_df['reward'].values.reshape(-1, 18)
    print('\nSupport:', rewards.shape[0])

    total_reward = np.mean(np.sum(rewards, axis=1))
    print('Empirical total reward of pi_b:', total_reward)
    
    
