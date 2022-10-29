import numpy as np
import pandas as pd


class IS:
    def __init__(self, behavior_policy_file, gamma=1.0, clip_degenerate=0):
        """ Implementation of the Stepwise Importance Sampling (IS) estimator.
            Please refer to https://arxiv.org/pdf/1807.01066.pdf for details.

            Params
            behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                  behavior policy, chosen actions ('action') and associated rewards ('reward').
            gamma:                Discount factor (default: 1.0)
            clip_degenerate:      For sharp behavior policies (e.g. softmax), weight degeneracy might result in
                                  unreliable estimates. Setting `clip_degenerate` will clip the top-N weights
                                  to their empirical mean (ClipIS), thereby reducing degeneracy.
                                  For details, see (Martino et al., 2018): https://ieeexplore.ieee.org/document/8450722
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
        self._clip_degenerate = clip_degenerate


    @staticmethod
    def _behavior_policy(df):
        # Return numeric columns assumed to belong to actions, i.e., '0'-'24'
        return df.filter(regex='\d+').values.astype(np.float64)

    def _to_table(self, arr):
        return arr.reshape(-1, self.timesteps)

    def _clip(self, weights):       
        for i in range(weights.shape[1]):
            # Determine threshold to mask top-k weights at each timestep
            sorted_weights = np.sort(weights, axis=0)
            thresholds = sorted_weights[-self._clip_degenerate]

            # Set weights > thres equal to their empirical mean at each timestep
            col_mask = weights[:, i] >= thresholds[:, i]
            weights[i, mask] = np.mean(weights[:, t])

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
        if self._clip_degenerate > 0:
            weights = self._clip(weights)

        if return_weights:
            return weights

        return np.mean(np.sum(self.gamma * weights * self.rewards, axis=1))


class WeightedIS(IS):
    def __init__(self, behavior_policy_file, gamma=1.0, clip_degenerate=0):
        """ Implementation of the Stepwise Weighted Importance Sampling (WIS) estimator.
            Please refer to https://arxiv.org/pdf/1807.01066.pdf for details.

            Params
            behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                  behavior policy, chosen actions ('action') and associated rewards ('reward').
            gamma:                Discount factor (default: 1.0)
            clip_degenerate:      For sharp behavior policies (e.g. softmax), weight degeneracy might result in
                                  unreliable estimates. Setting `clip_degenerate` will clip the top-N weights
                                  to their empirical mean (ClipIS), thereby reducing degeneracy.
                                  For details, see (Martino et al., 2018): https://ieeexplore.ieee.org/document/8450722
        """
        super().__init__(behavior_policy_file, gamma, clip_degenerate)

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
        ratio = np.cumprod(action_probs_e / action_probs_b, axis=1)
        if self._clip_degenerate > 0:
            ratio = self._clip(ratio)

        # Normalize weights
        weights = ratio / ratio.sum(axis=0, keepdims=True)
        if return_weights:
            return weights

        return np.sum(self.gamma * weights * self.rewards)


if __name__ == '__main__':
    np.set_printoptions(suppress=True)

    # Behavior policy
    behavior_policy_file = 'physician_policy/roggeveen_4h_with_cv/mimic-iii_valid_behavior_policy.csv'
    behavior_df = pd.read_csv(behavior_policy_file)
    behavior_policy = behavior_df.filter(regex='\d+').values  # -> 25 actions marked by integers 0 to 24!

    # Random policy
    from sklearn.preprocessing import label_binarize
    random_policy = label_binarize(np.random.randint(0, 24, behavior_policy.shape[0]), classes=np.arange(25))
    random_policy = random_policy.astype(np.float32) + np.random.uniform(0, 0.05, random_policy.shape)  # Add noise
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    # Zero-drug policy
    zerodrug_policy = np.full(behavior_policy.shape, fill_value=1e-12)
    zerodrug_policy[:, 0] = 1

    # IS/WIS + clipping
    imp_sampling = IS(behavior_policy_file)
    imp_clipped = IS(behavior_policy_file, clip_degenerate=2)
    
    wimp_sampling = WeightedIS(behavior_policy_file)
    wimp_clipped = WeightedIS(behavior_policy_file, clip_degenerate=2)

    print('IS:')
    print('behavior: ', imp_sampling(behavior_policy))
    print('random:   ', imp_sampling(random_policy))
    print('zero drug:', imp_sampling(zerodrug_policy))

    print('\nIS + clipping:')
    print('behavior: ', imp_clipped(behavior_policy))
    print('random:   ', imp_clipped(random_policy))
    print('zero drug:', imp_clipped(zerodrug_policy))

    print('\n', '-' * 32)

    print('\nWIS:')
    print('behavior: ', wimp_sampling(behavior_policy))
    print('random:   ', wimp_sampling(random_policy))
    print('zero drug:', wimp_sampling(zerodrug_policy))

    print('\nWIS + clipping:')
    print('behavior: ', wimp_clipped(behavior_policy))
    print('random:   ', wimp_clipped(random_policy))
    print('zero drug:', wimp_clipped(zerodrug_policy))

    print('\n', '-' * 32)

    # Sanity check: estimate true reward
    rewards = behavior_df['reward'].values.reshape(-1, 18)
    print('\nSupport:', rewards.shape[0])

    total_reward = np.mean(np.sum(rewards, axis=1))
    print('True total reward of pi_b:', total_reward)

    # Plot distribution of weights
    import matplotlib.pyplot as plt
    
    
