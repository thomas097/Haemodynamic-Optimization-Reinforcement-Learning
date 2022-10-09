import numpy as np
import pandas as pd


class WIS:
    def __init__(self, behavior_policy_file, gamma=0.9):
        """ Implementation of the Stepwise Weighted Importance Sampling (WIS) estimator.
            Please refer to https://arxiv.org/pdf/1807.01066.pdf for details.

            Params
            policy_file: DataFrame containing action probabilities (columns '0'-'24') for behavior policy,
                         chosen actions ('action') and associated rewards ('reward').
            gamma:       Discount factor
        """
        # Estimated action distribution of behavior policy (i.e. physician)
        df = pd.read_csv(behavior_policy_file)
        self._actions = np.expand_dims(df['action'], axis=1).astype(int)

        # Number of timesteps in episode
        self._timesteps = df.groupby('episode').size().max()

        # Reward function (drop terminal time steps without rewards)
        self._rewards = df['reward'].values.reshape(-1, self._timesteps)[:, :-1]
        self._gamma = gamma

        # Estimated action probabilities
        self._pi_b = self._behavior_policy(df)

    @staticmethod
    def _behavior_policy(df):
        # Return numeric columns (assumed to belong to actions, i.e., '0' - '24')
        action_cols = sorted([c for c in df.columns if c.isnumeric()], key=lambda c: int(c))
        return df[action_cols].values.astype(np.float64)

    def __call__(self, pi_e):
        """ Computes the WIS estimator for one episode (as marked by pi_b.index).

            pi_e:       Table of action probs acc. to πe with shape (num_states, num_actions)

            Returns:    Estimate of mean V^πe
        """
        # Extract table of action probabilities acc. to πe (πb) w.r.t. chosen actions
        pi_b = self._pi_b
        action_probs_e = np.take_along_axis(pi_e, self._actions, axis=1).reshape(-1, self._timesteps)
        action_probs_b = np.take_along_axis(pi_b, self._actions, axis=1).reshape(-1, self._timesteps)

        # Drop terminal time steps (have NaN rewards)
        action_probs_e = action_probs_e[:, :-1]
        action_probs_b = action_probs_b[:, :-1]

        # Compute importance ratio
        ratio = np.cumprod(action_probs_e / action_probs_b, axis=1)
        w = ratio / ratio.sum(axis=0, keepdims=True)

        # Compute gamma as a function of t
        gamma = np.power(self._gamma, np.arange(self._timesteps - 1))[np.newaxis]

        return np.sum(gamma * w * self._rewards)


if __name__ == '__main__':
    wis = WIS('physician_policy/mimic-iii_test_behavior_policy.csv')

    # Behavior policy
    behavior_policy = pd.read_csv('physician_policy/mimic-iii_test_behavior_policy.csv',
                                  usecols=[str(i) for i in range(25)]).values
    print('behavior:', wis(behavior_policy))

    # Random policy
    random_policy = np.random.uniform(0.0, 1, behavior_policy.shape)
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    print('random:  ', wis(random_policy))
