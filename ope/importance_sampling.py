import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
        self._actions = np.expand_dims(df['action'], axis=1).astype(int)

        # Estimated action probabilities
        self._pi_b = self._behavior_policy(df)

        # Number of timesteps in episode
        self._timesteps = df.groupby('episode').size().max()

        # Reward function (drop terminal time steps without rewards)
        self._rewards = df['reward'].values.reshape(-1, self._timesteps)
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
        action_probs_e = np.take_along_axis(pi_e, self._actions, axis=1).reshape(-1, self._timesteps)[:, :-1]
        action_probs_b = np.take_along_axis(pi_b, self._actions, axis=1).reshape(-1, self._timesteps)[:, :-1]

        # Compute cumulative importance ratio
        ratio = np.cumprod(action_probs_e / action_probs_b, axis=1)

        # Compute gamma as a function of t
        gamma = np.power(self._gamma, np.arange(self._timesteps - 1))[np.newaxis]

        # Drop NaN reward at terminal states
        rewards = self._rewards[:, :-1]

        return np.mean(np.sum(gamma * ratio * rewards, axis=1))


class WIS(IS):
    def __init__(self, behavior_policy_file, gamma=1.0):
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
        action_probs_e = np.take_along_axis(pi_e, self._actions, axis=1).reshape(-1, self._timesteps)[:, :-1]
        action_probs_b = np.take_along_axis(pi_b, self._actions, axis=1).reshape(-1, self._timesteps)[:, :-1]

        # Compute cumulative importance ratio
        ratio = np.cumprod(action_probs_e / action_probs_b, axis=1)
        w = ratio / ratio.sum(axis=0, keepdims=True)

        # Compute gamma as a function of t
        gamma = np.power(self._gamma, np.arange(self._timesteps - 1))[np.newaxis]

        # Drop NaN reward at terminal states
        rewards = self._rewards[:, :-1]

        return np.sum(gamma * w * rewards)


class GroupwiseWIS(IS):
    """ Implementation of the Groupwise Weighted Importance Sampling (WIS)
        estimator which accounts for confounding factors such as `disease severity`
        known to affect actions AND outcome variables (e.g. `base mortality rate`).

        GroupwiseWIS groups trajectories by some factor (e.g. `sirs_score`) and computes
        an average over the group-wise WIS estimates (weighted by their support).
    """
    def __init__(self, behavior_policy_file, dataset_file, group_by='sirs_score', gamma=1.0):
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
    behavior_df = pd.read_csv('physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv')
    behavior_policy = behavior_df[[str(i) for i in range(25)]].values  # assume 25 actions

    # Random policy
    random_policy = np.random.uniform(0.0, 1, behavior_policy.shape)
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    # Perfect policy
    def get_perfect_action(x):
        action1 = x * 0
        action1['1'] = 1
        return x if x['reward'].values[-2] == 15 else action1  # just behavior actions if R=15, else action 1
    perfect_policy = behavior_df.groupby('episode').apply(get_perfect_action)[[str(i) for i in range(25)]].values

    # Zero policy (inaction)
    zero_policy = np.zeros(behavior_policy.shape)
    zero_policy[:, 0] = 1

    is_ = IS('physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv')
    wis = WIS('physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv')
    gwis = GroupwiseWIS('physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv',
                        '../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv')

    print('IS:')
    print('behavior: ', is_(behavior_policy))
    print('perfect:  ', is_(perfect_policy))
    print('random:   ', is_(random_policy))
    print('zero:     ', is_(zero_policy))

    print('\nWIS:')
    print('behavior: ', wis(behavior_policy))
    print('perfect:  ', wis(perfect_policy))
    print('random:   ', wis(random_policy))
    print('zero:     ', wis(zero_policy))

    # print('\nGWIS:')
    # print('behavior: ', gwis(behavior_policy), gwis(behavior_policy, return_groups=True))
    # print('random:   ', gwis(random_policy), gwis(random_policy, return_groups=True))
    # print('foresight:', gwis(zero_policy), gwis(random_policy, return_groups=True))
    # print('zero:     ', gwis(zero_policy), gwis(zero_policy, return_groups=True))

    # Sanity check: estimate true reward
    rewards = behavior_df['reward'].values.reshape(-1, 18)[:, :-1]
    total_reward = np.mean(np.sum(rewards, axis=1))
    print('\nTrue total reward of pi_b:', total_reward)

    # Sanity check: estimate total reward of zero-drug trajectories
    actions = behavior_df['action'].values.reshape(-1, 18)[:, :-1]
    rewards = rewards[actions.max(axis=1) == 0]
    total_reward = np.mean(np.sum(rewards, axis=1))
    print('Expected zero-drug reward:', total_reward)
