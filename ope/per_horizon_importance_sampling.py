import numpy as np
import pandas as pd


class PHWIS:
    def __init__(self, behavior_policy_file):
        """ Implementation of Per-Horizon Weighted Importance Sampling (PHWIS) for Off-policy Evaluation (OPE)
        of treatment policies under trajectories with variable-length horizons.
        For details, please refer to https://www.ijcai.org/proceedings/2018/0729.pdf
        :param behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') under
                                     behavior policy, chosen actions ('action') and associated rewards ('reward').
        """
        behavior_df = pd.read_csv(behavior_policy_file)

        # determine probability of chosen action under behavior policy
        self._actions = behavior_df.action.values[:, np.newaxis].astype(int)
        probs_all = behavior_df.filter(regex='\d+').values
        probs_action = np.take_along_axis(probs_all, indices=self._actions, axis=1).flatten()

        # determine horizon of each episode
        horizon = behavior_df.groupby('episode').size()

        # count time steps in episode
        timesteps = behavior_df.groupby('episode').action.transform(lambda ep: np.arange(len(ep)))

        # combine into DataFrame
        self._df = pd.DataFrame({
            'episode': behavior_df.episode.astype(int),
            'reward': behavior_df.reward,
            'timestep': timesteps,
            'horizon': behavior_df.episode.apply(lambda x: horizon.loc[x]),
            'probs_b': probs_action
        })

        # keep track of effective sample size
        self._ess = []

    @property
    def ess(self):
        # weigh normalized importance weights by normalized horizon weight
        ess = np.array([1 / np.sum(w ** 2) for w in self._ess])
        return np.sum(ess)

    def ratios(self, pi_e, episodes=None):
        """ Returns a dataframe of cumulative importance ratios ('rho') for each timestep and episode
        :param pi_e:      Table of action probs acc. to πe with shape (num_states, num_actions)
        :param episodes:  Subset of episodes to consider in estimating V^πe
        :returns:         DataFrame with importance ratios and horizon weights
        """
        # compute importance ratio of action prob for chosen action under pi_e and pi_b
        df = self._df.copy()
        df['probs_e'] = np.take_along_axis(pi_e, indices=self._actions, axis=1).flatten()
        df['ratio'] = df['probs_e'] / df['probs_b']

        # filter episodes if episodes!=None
        if episodes is not None:
            df = df[df.episode.isin(episodes)]

        # compute importance weight, `rho`, at terminal timesteps and normalize over episodes of the same length
        df['cumprod_ratio'] = df.groupby('episode').ratio.cumprod()
        df['rho'] = df.groupby(['horizon', 'timestep']).cumprod_ratio.apply(lambda r: self._normalize(r))

        # compute weight of horizon, `wh`, as episodes_with_horizon / all_episodes
        df['wh'] = df.groupby('horizon').episode.transform(lambda x: x.nunique()) / df.episode.nunique()

        return df[['episode', 'rho', 'wh']]

    def _normalize(self, arr):
        """ Normalizes arr while taking care of divide-by-zero """
        total = np.sum(arr)
        if total == 0:
            return arr * 0 + (1 / np.prod(arr.shape))  # 1 / N
        return arr / total

    def __call__(self, pi_e, episodes=None):
        """ Computes the PHWIS estimate of V^πe
        :param pi_e:      Table of action probs acc. to πe with shape (num_states, num_actions)
        :param episodes:  Subset of episodes to consider in estimating V^πe
        :returns:         Estimate of mean V^πe
        """
        self._ess = []

        # create DataFrame with action prob for chosen action under pi_e and pi_b
        df = self._df.copy()
        df['probs_e'] = np.take_along_axis(pi_e, indices=self._actions, axis=1).flatten()

        # compute importance ratio
        df['ratio'] = df['probs_e'] / df['probs_b']

        # filter episodes if episodes!=None
        if episodes is not None:
            df = df[df.episode.isin(episodes)]

        # group episodes by horizon length
        wis, wh = [], []
        for horizon, horizon_df in df.groupby('horizon'):
            # compute normalized cumulative importance weights at terminal state
            # (where reward is issued) for episodes of same length
            ratios = horizon_df.ratio.values.reshape(-1, horizon)
            weights = self._normalize(np.prod(ratios ** 0.25, axis=1))

            # compute within-horizon WIS weight at terminal state
            rewards = horizon_df.reward.values[horizon-1::horizon]
            wis.append(np.sum(weights * rewards))

            # weigh horizon proportional to the number of episodes with this horizon
            n_episodes = weights.shape[0]
            wh.append(n_episodes)

            # keep track of weights to track sample size!
            self._ess.append(weights)

        # weigh WIS by horizon weights
        norm_wh = np.array(wh) / np.sum(wh)
        return np.array(wis).dot(norm_wh)


if __name__ == '__main__':
    # behavior policy
    behavior_policy_file = 'physician_policy/amsterdam-umc-db_aggregated_full_cohort_2h_knn/test_behavior_policy.csv'
    behavior_df = pd.read_csv(behavior_policy_file)
    behavior_policy = behavior_df.filter(regex='\d+').values  # -> 25 actions marked by integers 0 to 24!

    # policy that randomly deviates from physician 5% of the time
    semi_random_policy = behavior_policy.copy()
    i = np.random.random(semi_random_policy.shape[0]) < 0.05
    j = np.random.permutation(np.sum(i)) # shuffle rows
    semi_random_policy[i] = semi_random_policy[i][j]

    # random policy
    random_policy = np.random.uniform(0, 1, behavior_policy.shape)  # Noise
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    estimator = PHWIS(behavior_policy_file)
    print('PHWIS:')
    behavior_phwis = estimator(behavior_policy)
    print('Behavior policy: %.3f (%.2f)' % (behavior_phwis, estimator.ess))

    semi_random_phwis = estimator(semi_random_policy)
    print('Semi-random policy: %.3f (%.2f)' % (semi_random_phwis, estimator.ess))

    random_phwis = estimator(random_policy)
    print('Random policy: %.3f (%.2f)' % (random_phwis, estimator.ess))

    # Sanity unbiased w.r.t. behavior policy?
    rewards = behavior_df.reward.values
    exp_reward = np.mean(rewards[rewards != 0])
    print('\nTrue reward behavior policy:', exp_reward)
    print('Support:', behavior_df.episode.nunique())


    
    
