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
        all_timesteps = df.episode.value_counts().values
        self.timesteps = all_timesteps[0]
        if np.any(all_timesteps != self.timesteps):
            raise exception('Episodes must be of the same length. Found %s' % np.unique(all_timesteps))

        # behavior policy's actions and action probabilities
        self.actions = np.expand_dims(df['action'], axis=1).astype(int)
        self.pi_b = self._behavior_policy(df)

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
    def __init__(self, behavior_policy_file, gamma=1.0, bootstraps=0, conf=0.95):
        """ Implementation of Stepwise Weighted Importance Sampling (WIS) for Off-policy Policy Evaluation (OPE)
        Optionally the method uses a non-parametric bootstrap to obtain a lower/upper confidence bound
        Please refer to https://arxiv.org/pdf/1807.01066.pdf for details
        :param behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                  behavior policy, chosen actions ('action') and associated rewards ('reward').
        :param gamma:                Discount factor (default: 1.0)
        :param bootstraps:           How many bootstrap sets to use to estimate confidence bounds (default: 0; disabled)
        :param conf:                 Confidence level (CL) of the bootstrap estimate (default: CL 95%)
        """
        super().__init__(behavior_policy_file, gamma=gamma, bootstraps=bootstraps, conf=conf)
        self._ess = None

    @property
    def effective_sample_size(self):
        return self._ess

    def _compute_effective_sample_size(self, ratios):
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

    def __call__(self, pi_e, return_ratios=False):
        """ Computes the WIS estimate of V^πe
        :param pi_e:  Table of action probs acc. to πe with shape (num_states, num_actions)
        :returns:     Estimate of mean V^πe. If bootstraps > 0, a tuple (lower, median, upper) is returned
        """
        # extract action probabilities acc. to πe (πb) w.r.t. chosen actions
        action_probs_e = self._to_table(np.take_along_axis(pi_e, self.actions, axis=1))
        action_probs_b = self._to_table(np.take_along_axis(self.pi_b, self.actions, axis=1))

        # compute cumulative importance ratio
        ratios = np.cumprod(action_probs_e / action_probs_b, axis=1)

        # compute ESS for later reference
        self._ess = self._compute_effective_sample_size(ratios)

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
    behavior_policy_file = 'physician_policy/aggregated_4h/mimic-iii_test_behavior_policy.csv'
    behavior_df = pd.read_csv(behavior_policy_file)
    behavior_policy = behavior_df.filter(regex='\d+').values  # -> 25 actions marked by integers 0 to 24!

    # random policy
    random_policy = np.random.uniform(0, 1, behavior_policy.shape)  # Noise
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    # zero-drug policy
    zerodrug_policy = np.full(behavior_policy.shape, fill_value=1e-6)
    zerodrug_policy[:, 0] = 1 - (24 * 1e-6)

    # highest IV/VP policy
    aggressive_policy = np.full(behavior_policy.shape, fill_value=1e-6)
    aggressive_policy[:, 24] = 1 - (24 * 1e-6)

    # estimate confidence intervals over WIS scores using 1000 bootstrap sets
    wimp_sampling = WeightedIS(behavior_policy_file, bootstraps=1000)
    conf_intervals = np.array([
        wimp_sampling(behavior_policy),
        wimp_sampling(random_policy),
        wimp_sampling(zerodrug_policy),
        wimp_sampling(aggressive_policy)
    ])

    # Plot confidence bounds as errorplot
    import matplotlib.pyplot as plt

    plt.figure()
    lower_err = conf_intervals[:, 1] - conf_intervals[:, 0]
    upper_err = conf_intervals[:, 2] - conf_intervals[:, 1]
    plt.errorbar(x=[1, 2, 3, 4], y=conf_intervals[:, 1], yerr=[lower_err, upper_err], fmt='o', color='k')
    plt.xticks(ticks=[1, 2, 3, 4], labels=['physician', 'random', 'zero-drug', 'aggressive'])
    plt.xlabel('policy')
    plt.ylabel('WIS')
    plt.ylim(-16, 16)
    plt.grid(True)
    plt.show()

    # Sanity check: estimate true reward
    timesteps = behavior_df.episode.value_counts().max()
    rewards = behavior_df.reward.values.reshape(-1, timesteps)
    print('\nSupport:', rewards.shape[0])

    total_reward = np.mean(np.sum(rewards, axis=1))
    print('Empirical total reward of pi_b:', total_reward)

    # Sanity check: Effective sample size as a function of deviance from physician policy
    from tqdm import tqdm
    wimp_sampling = WeightedIS(behavior_policy_file, bootstraps=0)

    dev, wis, ess = [], [], []
    for p in tqdm(np.linspace(0, 1, 100)):
        wis_scores = []
        ess_scores = []
        for _ in range(75):
            # assign a random action distribution to `p` percentage of actions of evaluation policy!
            mask = np.random.random(behavior_policy.shape[0]) < p
            perm = np.random.permutation(behavior_policy.shape[1])
            randomized_policy = np.copy(behavior_policy)
            randomized_policy[mask] = randomized_policy[mask][:, perm] # shuffle action prob columns

            # ensure actions receive non-zero probability
            randomized_policy = (randomized_policy + 1e-5) / np.sum(randomized_policy + 1e-5, axis=1, keepdims=True)

            # run WIS and get estimate of sample size
            wis_scores.append(wimp_sampling(randomized_policy))
            ess_scores.append(wimp_sampling.effective_sample_size)

        dev.append(p)
        wis.append(np.mean(wis_scores))
        ess.append(np.mean(ess_scores))

    plt.figure()
    plt.subplot(121)
    plt.plot(dev, wis)
    plt.title('WIS scores')
    plt.xlabel('% actions deviating from physician')
    plt.ylabel('WIS score')
    plt.subplot(122)
    plt.plot(dev, ess)
    plt.title('Effective sample size')
    plt.xlabel('% actions deviating from physician')
    plt.ylabel('Effective sample size')
    plt.show()


    
    
