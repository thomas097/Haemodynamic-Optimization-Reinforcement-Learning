import numpy as np
import pandas as pd


class IS:
    def __init__(self, behavior_policy_file, gamma=1.0, bootstraps=0, conf=0.95):
        """ Implementation of Stepwise Importance Sampling (IS) for Off-policy Policy Evaluation (OPE).
        Optionally the method uses a non-parametric bootstrap to obtain a confidence bound on the estimate.
        Please refer to https://arxiv.org/pdf/1807.01066.pdf for details.
        :param behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                     behavior policy, chosen actions ('action') and associated rewards ('reward').
        :param gamma:                Discount factor (default: 1.0)
        :param bootstraps:           How many bootstrap sets to use to estimate confidence bounds (default: 0; disabled)
        :param conf:                 Confidence level (CL) of the bootstrap estimate (default: CL 95%)
        """
        behavior_df = pd.read_csv(behavior_policy_file)

        self._n_timesteps = self._length_of_trajectories(behavior_df)
        self._n_episodes = len(behavior_df.episode.unique())

        self._pi_b = self._get_behavior_policy(behavior_df)
        self._actions = behavior_df.action.values[:, np.newaxis].astype(int)
        self._rewards = behavior_df.reward.values.reshape(-1, self._n_timesteps)

        # precompute discounting factor
        self._gammas = np.power(gamma, np.arange(self._n_timesteps))[np.newaxis]
        self.ratios = None

        # bootstrapping parameters
        self._bootstraps = bootstraps
        self._conf = conf

    @staticmethod
    def _length_of_trajectories(behavior_df):
        """ Infers episode length, or horizon, from behavior policy file
        :param behavior_df:  pandas DataFrame of behavior policy file
        :return:             Number of timesteps in episodes
        """
        timesteps = behavior_df.episode.value_counts().values
        if np.any(timesteps != timesteps[0]):
            raise Exception('variable horizon episodes found')
        return timesteps[0]

    @staticmethod
    def _get_behavior_policy(behavior_df):
        """  Infers columns corresponding to the action probabilities at each state
        :param df:  see documentation for `_horizon`
        """
        # Return numeric columns assumed to belong to actions, i.e., '0'-'24'
        return behavior_df.filter(regex='\d+').values.astype(np.float64)

    def __call__(self, pi_e):
        """ Computes the IS estimate of V^πe
        :param pi_e:  Table of action probs acc. to πe with shape (num_states, num_actions)
        :returns:     Estimate of mean V^πe. If bootstraps > 0, a tuple (lower, median, upper) is returned
        """
        # extract table of action probabilities acc. to πe (πb) w.r.t. chosen actions
        pi_b = self._pi_b
        action_probs_e = np.take_along_axis(pi_e, self._actions, axis=1).reshape(-1, self._n_timesteps)
        action_probs_b = np.take_along_axis(pi_b, self._actions, axis=1).reshape(-1, self._n_timesteps)

        # compute cumulative importance ratio
        self.ratios = np.cumprod(action_probs_e / action_probs_b, axis=1)

        # if bootstraps enabled, estimate distribution of scores by bootstrapping
        if self._bootstraps > 0:
            estimates = []
            for _ in range(self._bootstraps):
                # sample with replacement a bootstrap set of size |episodes|
                i = np.random.choice(self._n_episodes, size=self._n_episodes, replace=True)
                v_pi = np.mean(np.sum(self._gammas * self.ratios[i] * self._rewards[i], axis=1))
                estimates.append(v_pi)

            median = np.median(estimates)
            lower = np.quantile(estimates, q=(1 - self._conf) / 2)
            upper = np.quantile(estimates, q=(1 + self._conf) / 2)
            return lower, median, upper

        # else; estimate over the complete test set
        return np.sum(self._gammas * self.ratios * self._rewards, axis=1)


class WIS(IS):
    def __init__(self, behavior_policy_file, gamma=1.0, bootstraps=0, conf=0.95):
        """ Implementation of Stepwise Weighted Importance Sampling (WIS) for Off-policy Policy Evaluation (OPE)
        Optionally the method uses a non-parametric bootstrap to obtain a confidence interval over the estimate
        Please refer to https://arxiv.org/pdf/1807.01066.pdf for details
        :param behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                     behavior policy, chosen actions ('action') and associated rewards ('reward').
        :param gamma:                Discount factor (default: 1.0)
        :param bootstraps:           How many bootstrap sets to use to estimate confidence bounds (default: 0; disabled)
        :param conf:                 Confidence level (CL) of the bootstrap estimate (default: CL 95%)
        """
        super().__init__(behavior_policy_file, gamma=gamma, bootstraps=bootstraps, conf=conf)
        self.ess = None

    def _compute_effective_sample_size(self, ratios, timestep=-1):
        """ Estimates effective sample size of estimator
        See: http://www.nowozin.net/sebastian/blog/effective-sample-size-in-importance-sampling.html
        :param ratios:     Unnormalized importance ratios
        :param timestep:   At which timestep to evaluate sample size
        :return:           Effective sample size
        """
        norm_ratios = ratios / ratios.sum(axis=0, keepdims=True)
        return 1 / np.sum(norm_ratios[:, timestep] ** 2)

    def __call__(self, pi_e):
        """ Computes the WIS estimate of V^πe
        :param pi_e:  Table of action probs acc. to πe with shape (num_states, num_actions)
        :returns:     Estimate of mean V^πe. If bootstraps > 0, a tuple (lower, median, upper) is returned
        """
        # extract action probabilities acc. to πe (πb) w.r.t. chosen actions
        pi_b = self._pi_b
        action_probs_e = np.take_along_axis(pi_e, self._actions, axis=1).reshape(-1, self._n_timesteps)
        action_probs_b = np.take_along_axis(pi_b, self._actions, axis=1).reshape(-1, self._n_timesteps)

        # compute cumulative importance ratio
        ratios = np.cumprod(action_probs_e / action_probs_b, axis=1)
        self.ratios = ratios

        # compute ESS for later reference
        self.ess = self._compute_effective_sample_size(ratios)

        # bootstrapping
        if self._bootstraps > 0:
            estimates = []
            # estimate v_pi over N bootstrap sets
            for _ in range(self._bootstraps):
                i = np.random.choice(self._n_episodes, size=self._n_episodes, replace=True)
                norm_ratios = ratios[i] / ratios[i].sum(axis=0, keepdims=True)
                v_pi = np.sum(self._gammas * norm_ratios * self._rewards[i])
                estimates.append(v_pi)

            # estimate e.g. CL-confidence interval
            median = np.median(estimates)
            lower = np.quantile(estimates, q=(1 - self._conf) / 2)
            upper = np.quantile(estimates, q=(1 + self._conf) / 2)
            return lower, median, upper

        # else; returns estimate over the complete test set
        norm_ratios = self.ratios / self.ratios.sum(axis=0, keepdims=True)
        return np.sum(self._gammas * norm_ratios * self._rewards)


if __name__ == '__main__':
    # behavior policy
    behavior_policy_file = 'physician_policy/aggregated_all_1h/mimic-iii_valid_behavior_policy.csv'
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
    wimp_sampling = WIS(behavior_policy_file, bootstraps=1000)
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
    plt.ylim(-18, 18)
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
    wimp_sampling = WIS(behavior_policy_file, bootstraps=0)

    dev, wis, ess = [], [], []
    for p in tqdm(np.linspace(0, 1, 100)):
        wis_scores = []
        ess_scores = []
        for _ in range(50):
            # assign a random action distribution to `p` percentage of actions of evaluation policy!
            mask = np.random.random(behavior_policy.shape[0]) < p
            perm = np.random.permutation(behavior_policy.shape[1])
            randomized_policy = np.copy(behavior_policy)
            randomized_policy[mask] = randomized_policy[mask][:, perm] # shuffle action prob columns

            # ensure actions receive non-zero probability
            randomized_policy = (randomized_policy + 1e-5) / np.sum(randomized_policy + 1e-5, axis=1, keepdims=True)

            # run WIS and get estimate of sample size
            wis_scores.append(wimp_sampling(randomized_policy))
            ess_scores.append(wimp_sampling.ess)

        dev.append(p)
        wis.append(np.mean(wis_scores))
        ess.append(np.mean(ess_scores))

    plt.figure()
    plt.subplot(121)
    plt.plot(dev, wis)
    plt.ylim(-15, 15)
    plt.title('WIS scores')
    plt.xlabel('% actions deviating from physician')
    plt.ylabel('WIS score')
    plt.subplot(122)
    plt.plot(dev, ess)
    plt.title('Effective sample size')
    plt.xlabel('% actions deviating from physician')
    plt.ylabel('Effective sample size')
    plt.show()


    
    
