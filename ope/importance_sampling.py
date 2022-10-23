import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize


class IS:
    def __init__(self, behavior_policy_file, gamma=1.0):
        """ Implementation of the Stepwise Importance Sampling (IS) estimator.
            Please refer to https://arxiv.org/pdf/1807.01066.pdf for details.

            Params
            behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                  behavior policy, chosen actions ('action') and associated rewards ('reward').
            gamma:                Discount factor
        """
        # Actions taken by behavior policy (i.e. physician)
        df = pd.read_csv(behavior_policy_file)
        self.actions = np.expand_dims(df['action'], axis=1).astype(int)

        # Estimated action probabilities
        self.pi_b = self._behavior_policy(df)

        # Number of timesteps in episode
        self.timesteps = df.groupby('episode').size().max()

        # Reward function (drop terminal time steps without rewards)
        self.rewards = self._to_table(df['reward'].values)
        self.gamma = gamma

    @staticmethod
    def _behavior_policy(df):
        # Return numeric columns assumed to belong to actions, i.e., '0'-'24'
        return df.filter(regex='\d+').values.astype(np.float64)

    def _to_table(self, arr):
        return np.nan_to_num(arr.reshape(-1, self.timesteps), nan=0.0)  # Replace NaNs in reward table

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
            return ratio

        # Compute gamma as a function of t
        gamma = np.power(self.gamma, np.arange(self.timesteps))[np.newaxis]

        return np.mean(np.sum(gamma * weights * self.rewards, axis=1))


class WeightedIS(IS):
    def __init__(self, behavior_policy_file, gamma=1.0):
        """ Implementation of the Stepwise Weighted Importance Sampling (WIS) estimator.
            Please refer to https://arxiv.org/pdf/1807.01066.pdf for details.

            Params
            behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                  behavior policy, chosen actions ('action') and associated rewards ('reward').
            gamma:                Discount factor
        """
        super().__init__(behavior_policy_file, gamma)

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
        weights = ratio / (ratio.sum(axis=0, keepdims=True) + 1e-12)
        if return_weights:
            return weights

        # Compute gamma as a function of t
        gamma = np.power(self.gamma, np.arange(self.timesteps))[np.newaxis]

        return np.sum(gamma * weights * self.rewards)


if __name__ == '__main__':
    # Behavior policy
    behavior_df = pd.read_csv('physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv')
    behavior_policy = behavior_df.filter(regex='\d+').values  # -> 25 actions marked by integers 0 to 24!

    # Random policy
    random_policy = label_binarize(np.random.randint(0, 24, behavior_policy.shape[0]), classes=np.arange(25))
    random_policy = random_policy.astype(np.float32) + np.random.uniform(0, 0.05, random_policy.shape) # Add noise
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    # Bad policy (high likelihood of action 1 which is not possible)
    bad_policy = np.ones(behavior_policy.shape)
    bad_policy[:, 1] = 50
    bad_policy = bad_policy / np.sum(bad_policy, axis=1, keepdims=True)

    # Zero-drug policy
    zerodrug_policy = np.full(behavior_policy.shape, fill_value=1e-12)
    zerodrug_policy[:, 0] = 1

    imp_sampling = IS('physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv')
    wimp_sampling = WeightedIS('physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv')

    print('IS:')
    print('behavior: ', imp_sampling(behavior_policy))
    print('random:   ', imp_sampling(random_policy))
    print('bad:      ', imp_sampling(bad_policy))
    print('zero drug:', imp_sampling(zerodrug_policy))

    print('\nWIS:')
    print('behavior: ', wimp_sampling(behavior_policy))
    print('random:   ', wimp_sampling(random_policy))
    print('bad:      ', wimp_sampling(bad_policy))
    print('zero drug:', wimp_sampling(zerodrug_policy))

    # Sanity check: estimate true reward
    rewards = np.nan_to_num(behavior_df['reward'].values.reshape(-1, 18), nan=0.0)
    print('\nSupport:', rewards.shape[0])

    total_reward = np.mean(np.sum(rewards, axis=1))
    print('True total reward of pi_b:', total_reward)