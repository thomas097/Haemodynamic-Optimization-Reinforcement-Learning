import numpy as np
import pandas as pd


class IS:
    def __init__(self, behavior_policy_file, gamma=1.0):
        """ Implementation of the Stepwise Importance Sampling (IS) estimator.
            Please refer to https://arxiv.org/pdf/1807.01066.pdf for details.

            Params
            behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                  behavior policy, chosen actions ('action') and associated rewards ('reward').
            gamma:                Discount factor (default: 1.0)
        """
        phys_df = pd.read_csv(behavior_policy_file)

        # Actions taken by behavior policy (i.e. physician)
        self.actions = np.expand_dims(phys_df['action'], axis=1).astype(int)

        # Estimated action probabilities
        self.pi_b = self._behavior_policy(phys_df)

        # Reward and discounting
        self.timesteps = phys_df.groupby('episode').size().max()
        self.rewards = self._to_table(phys_df['reward'].values)
        self.gamma = np.power(gamma, np.arange(self.timesteps))[np.newaxis]

    @staticmethod
    def _behavior_policy(df):
        # Return numeric columns assumed to belong to actions, i.e., '0'-'24'
        return df.filter(regex='\d+').values.astype(np.float64)

    def _to_table(self, arr):
        return arr.reshape(-1, self.timesteps)

    def _clip(self, weights):
        # Clip each row of weight matrix separately
        for i in range(weights.shape[1]):
            # Determine threshold to mask top-k weights at timestep i
            threshold = np.sort(weights[:, i])[-self._min_samples]

            # Set weights > threshold equal to their empirical mean
            mask = weights[:, i] >= threshold
            weights[mask, i] = np.mean(weights[mask, i])

        return weights

    def __call__(self, pi_e, return_weights=False):
        """ Computes the IS estimate of V^πe.

            pi_e:       Table of action probs acc. to πe with shape (num_states, num_actions)

            Returns:    Estimate of mean V^πe
        """
        # Extract table of action probabilities acc. to πe (πb) w.r.t. chosen actions
        pi_b = self.pi_b
        action_probs_e = self._to_table(np.take_along_axis(pi_e, self.actions, axis=1))
        action_probs_b = self._to_table(np.take_along_axis(pi_b, self.actions, axis=1))

        # Compute cumulative importance ratio
        weights = np.cumprod(action_probs_e / action_probs_b, axis=1)
        if return_weights:
            return weights

        return np.mean(np.sum(self.gamma * weights * self.rewards, axis=1))


class WeightedIS(IS):
    def __init__(self, behavior_policy_file, gamma=1.0, verbose=False):
        """ Implementation of the Stepwise Weighted Importance Sampling (WIS) estimator.
            Please refer to https://arxiv.org/pdf/1807.01066.pdf for details.

            Params
            behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                  behavior policy, chosen actions ('action') and associated rewards ('reward').
            gamma:                Discount factor (default: 1.0)
            verbose:              Whether to print information such as effective sample size (ESS) to stdout.
        """
        super().__init__(behavior_policy_file, gamma)
        self.verbose = verbose

    def __call__(self, pi_e, return_weights=False):
        """ Computes the WIS estimate of V^πe.

            pi_e:       Table of action probs acc. to πe with shape (num_states, num_actions)

            Returns:    Estimate of mean V^πe
        """
        # Extract action probabilities acc. to πe (πb) w.r.t. chosen actions
        pi_b = self.pi_b
        action_probs_e = self._to_table(np.take_along_axis(pi_e, self.actions, axis=1))
        action_probs_b = self._to_table(np.take_along_axis(pi_b, self.actions, axis=1))

        # Compute cumulative importance ratio
        ratio = np.cumprod(action_probs_e / action_probs_b, axis=1) ** (1 / 2)

        # Normalize weights
        weights = ratio / ratio.sum(axis=0, keepdims=True)
        if return_weights:
            return weights

        if self.verbose:
            # See: http://www.nowozin.net/sebastian/blog/effective-sample-size-in-importance-sampling.html
            ess = 1 / np.sum(weights[:, -1] ** 2)
            print('\nEffective sample size: %.1f' % ess)

        return np.sum(self.gamma * weights * self.rewards)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    # Behavior policy
    behavior_policy_file = 'physician_policy/roggeveen_4h_with_cv/mimic-iii_valid_behavior_policy.csv'
    behavior_df = pd.read_csv(behavior_policy_file)
    behavior_policy = behavior_df.filter(regex='\d+').values  # -> 25 actions marked by integers 0 to 24!

    # Dr. House policy (mimics the physician if patient survives (i.e. does the right thing), but goes crazy otherwise)
    from sklearn.preprocessing import label_binarize
    house_policy = label_binarize(behavior_df.action, classes=np.arange(25))
    crazy_action = label_binarize([1], classes=np.arange(25))
    for i, r in enumerate(behavior_df.reward):
        if r == -15:
            house_policy[i-17:i + 1] = crazy_action  # crazy_action is impossible (VP but no IV)

    # Random policy
    random_policy = np.random.uniform(0, 1, behavior_policy.shape)  # Add noise
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    # Zero-drug policy
    zerodrug_policy = np.full(behavior_policy.shape, fill_value=1e-6)
    zerodrug_policy[:, 0] = 1 - 24e-6

    # IS/WIS
    imp_sampling = IS(behavior_policy_file)
    wimp_sampling = WeightedIS(behavior_policy_file, verbose=True)

    print('IS:')
    print('behavior: ', imp_sampling(behavior_policy))
    print('Dr. House:', imp_sampling(house_policy))
    print('random:   ', imp_sampling(random_policy))
    print('zero drug:', imp_sampling(zerodrug_policy))

    print('\n', '-' * 32)

    print('\nWIS:')
    print('behavior: ', wimp_sampling(behavior_policy))
    print('Dr. House:', wimp_sampling(house_policy))
    print('random:   ', wimp_sampling(random_policy))
    print('zero drug:', wimp_sampling(zerodrug_policy))

    print('\n', '-' * 32)

    # Sanity check: estimate true reward
    rewards = behavior_df['reward'].values.reshape(-1, 18)
    print('\nSupport:', rewards.shape[0])

    total_reward = np.mean(np.sum(rewards, axis=1))
    print('Empirical total reward of pi_b:', total_reward)
    
    
