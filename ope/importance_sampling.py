import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize


class IS:
    def __init__(self, behavior_policy_file, gamma=0.9):
        """ Implementation of the Stepwise Importance Sampling (IS) estimator.
            Please refer to https://arxiv.org/pdf/1807.01066.pdf for details.

            Params
            behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                  behavior policy, chosen actions ('action') and associated rewards ('reward').
            gamma:                Discount factor
        """
        # Actions taken by behavior policy (i.e. physician)
        df = pd.read_csv(behavior_policy_file)
        self._actions = np.expand_dims(df['action'], axis=1).astype(int)

        # Estimated action probabilities
        self._pi_b = self._behavior_policy(df)

        # Number of timesteps in episode
        self._timesteps = df.groupby('episode').size().max()

        # Reward function (drop terminal time steps without rewards)
        self._rewards = df['reward'].values.reshape(-1, self._timesteps)[:, :-1]
        self._gamma = gamma

    @staticmethod
    def _behavior_policy(df):
        # Return numeric columns (assumed to belong to actions, i.e., '0' - '24')
        action_cols = sorted([c for c in df.columns if c.isnumeric()], key=lambda c: int(c))
        return df[action_cols].values.astype(np.float64)

    def __call__(self, pi_e):
        """ Computes the IS estimate of V^πe.

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

        # Compute cumulative importance ratio
        w = np.cumprod(action_probs_e / action_probs_b, axis=1)

        # Compute gamma as a function of t
        gamma = np.power(self._gamma, np.arange(self._timesteps - 1))[np.newaxis]

        return np.mean(np.sum(gamma * w * self._rewards, axis=1))


class WIS(IS):
    def __init__(self, behavior_policy_file, gamma=0.9):
        """ Implementation of the Stepwise Weighted Importance Sampling (WIS) estimator.
            Please refer to https://arxiv.org/pdf/1807.01066.pdf for details.

            Params
            behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                  behavior policy, chosen actions ('action') and associated rewards ('reward').
            gamma:                Discount factor
        """
        super().__init__(behavior_policy_file, gamma)

    def __call__(self, pi_e):
        """ Computes the WIS estimate of V^πe.

            pi_e:       Table of action probs acc. to πe with shape (num_states, num_actions)

            Returns:    Estimate of mean V^πe
        """
        # Extract action probabilities acc. to πe (πb) w.r.t. chosen actions
        pi_b = self._pi_b
        action_probs_e = np.take_along_axis(pi_e, self._actions, axis=1).reshape(-1, self._timesteps)
        action_probs_b = np.take_along_axis(pi_b, self._actions, axis=1).reshape(-1, self._timesteps)

        # Drop terminal time steps (no reward received in these states)
        action_probs_e = action_probs_e[:, :-1]
        action_probs_b = action_probs_b[:, :-1]

        # Compute cumulative importance ratio
        ratio = np.cumprod(action_probs_e / action_probs_b, axis=1)
        w = ratio / ratio.sum(axis=0, keepdims=True)

        # Compute gamma as a function of t
        gamma = np.power(self._gamma, np.arange(self._timesteps - 1))[np.newaxis]

        return np.sum(gamma * w * self._rewards)


class GWIS(IS):
    """ Implementation of the Groupwise Weighted Importance Sampling (GWIS)
        estimator which accounts for confounding factors such as `disease severity`
        known to affect actions AND outcome variables (e.g. `base mortality rate`).

        GWIS groups trajectories by some factor (e.g. `sirs_score`) and computes
        an average over group-wise WIS estimates (weighted by their support!).
    """
    def __init__(self, behavior_policy_file, dataset_file, group_by='sirs_score', gamma=0.9):
        super().__init__(behavior_policy_file, gamma)
        # Assign indices to each group in `group_by`
        groups = pd.read_csv(dataset_file, usecols=[group_by])[group_by].values
        self._group_indices = [np.array(groups == i) for i in sorted(set(groups))]

        # Compute contribution of each group
        self._support = np.array([np.sum(idx) for idx in self._group_indices]) / groups.shape[0]

        # Overwrite: rewards (do not drop last terminal state yet!)
        self._rewards = pd.read_csv(behavior_policy_file, usecols=['reward'])['reward'].values

    def __call__(self, pi_e, return_groups=False):
        """ Computes the WIS estimate of V^πe.

            pi_e:       Table of action probs acc. to πe with shape (num_states, num_actions)

            Returns:    Estimate of mean V^πe
        """
        # To collect WIS scores for each group
        wis_scores = np.zeros(len(self._group_indices), dtype=np.float64)

        # Consider groups one by one
        for i, idx in enumerate(self._group_indices):
            # Extract action probabilities acc. to πe (πb) w.r.t. chosen actions
            pi_b = self._pi_b
            action_probs_e = np.take_along_axis(pi_e[idx], self._actions[idx], axis=1).reshape(-1, self._timesteps)
            action_probs_b = np.take_along_axis(pi_b[idx], self._actions[idx], axis=1).reshape(-1, self._timesteps)

            # Create table of rewards
            rewards = self._rewards[idx].reshape(-1, self._timesteps)

            # Drop terminal time steps (NaN rewards received in these states)
            action_probs_e = action_probs_e[:, :-1]
            action_probs_b = action_probs_b[:, :-1]
            rewards = rewards[:, :-1]

            # Compute cumulative importance ratio
            ratio = np.cumprod(action_probs_e / action_probs_b, axis=1)
            w = ratio / ratio.sum(axis=0, keepdims=True)

            # Compute within-group WIS
            gamma = np.power(self._gamma, np.arange(self._timesteps - 1))[np.newaxis]
            wis_scores[i] = np.sum(gamma * w * rewards)

        if return_groups:
            return wis_scores

        # Average WIS scores by their support
        return wis_scores.dot(self._support)


if __name__ == '__main__':
    # Behavior policy
    behavior_df = pd.read_csv('physician_policy/mimic-iii_train_behavior_policy.csv')
    behavior_policy = behavior_df[[str(i) for i in range(25)]].values

    # Random policy
    random_policy = np.random.uniform(0.0, 1, behavior_policy.shape)
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    # Zero policy (inaction)
    zero_policy = np.zeros(behavior_policy.shape)
    zero_policy[:, 0] = 1

    is_ = IS('physician_policy/mimic-iii_train_behavior_policy.csv')
    wis = WIS('physician_policy/mimic-iii_train_behavior_policy.csv')
    gwis = GWIS('physician_policy/mimic-iii_train_behavior_policy.csv',
                '../preprocessing/datasets/mimic-iii/roggeveen/mimic-iii_train.csv')

    print('IS:')
    print('behavior:', is_(behavior_policy))
    print('random:  ', is_(random_policy))
    print('zero:    ', is_(zero_policy))

    print('\nWIS:')
    print('behavior:', wis(behavior_policy))
    print('random:  ', wis(random_policy))
    print('zero:    ', wis(zero_policy))

    print('\nGWIS:')
    print('behavior:', gwis(behavior_policy, return_groups=True))
    print('random:  ', gwis(random_policy, return_groups=True))
    print('zero:    ', gwis(zero_policy, return_groups=True))

    # Sanity check: estimate true discounted reward
    rewards = behavior_df['reward'].values.reshape(-1, 18)[:, :-1]
    gamma = np.power(0.9, np.arange(17))[np.newaxis]
    discounted_reward = np.mean(np.sum(gamma * rewards, axis=1))
    print('\nTrue discounted reward of pi_b:', discounted_reward)
