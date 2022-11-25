import numpy as np
import pandas as pd


class PHWIS:
    def __init__(self, behavior_policy_file):
        """ Implementation of Per-Horizon Weighted Importance Sampling (PHWIS) for Off-policy Evaluation (OPE)
        of treatment policies under patient trajectories with variable horizons.
        Please refer to https://www.ijcai.org/proceedings/2018/0729.pdf for details
        :param behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                     behavior policy, chosen actions ('action') and associated rewards ('reward').
        """
        behavior_df = pd.read_csv(behavior_policy_file)

        # determine probability of chosen action under behavior policy
        self._actions = np.expand_dims(behavior_df.action.values, axis=1).astype(int)
        probs_all = behavior_df.filter(regex='\d+').values
        probs_action = np.take_along_axis(probs_all, indices=self._actions, axis=1).flatten()

        # determine horizon of each episode
        horizon = behavior_df.groupby('episode').size()

        # compute timestep in episode
        timesteps = behavior_df.groupby('episode').action.transform(lambda ep: np.arange(len(ep)))

        # combine into DataFrame
        self._df = pd.DataFrame({
            'episode': behavior_df.episode.astype(int),
            'reward': behavior_df.reward,
            'timestep': timesteps,
            'horizon': behavior_df.episode.apply(lambda x: horizon.loc[x]),
            'probs_b': probs_action
        })

    def ratios(self, probs_action):
        """ Returns a dataframe of cumulative importance ratios ('rho') for each timestep and episode
        """
        # compute importance ratio of action prob for chosen action under pi_e and pi_b
        df = self._df.copy()
        df['probs_e'] = np.take_along_axis(probs_action, indices=self._actions, axis=1).flatten()
        df['ratio'] = df['probs_e'] / df['probs_b']

        # compute importance weight, `rho`, at terminal timesteps and normalize over episodes of the same length
        df['cumprod_ratio'] = df.groupby('episode').ratio.cumprod()
        df['rho'] = df.groupby(['horizon', 'timestep']).cumprod_ratio.apply(lambda r: r / r.sum())

        # compute weight of horizon, `wh`, as episodes_with_horizon / all_episodes
        df['wh'] = df.groupby('horizon').episode.transform(lambda x: x.nunique()) / df.episode.nunique()

        return df[['episode', 'rho', 'wh']]

    def __call__(self, probs_action):
        # create DataFrame with action prob for chosen action under pi_e and pi_b
        df = self._df.copy()
        df['probs_e'] = np.take_along_axis(probs_action, indices=self._actions, axis=1).flatten()

        # compute importance ratio
        df['ratio'] = df['probs_e'] / df['probs_b']

        # group episodes by horizon length
        wis, wh = [], []
        for horizon, horizon_df in df.groupby('horizon'):
            # compute normalized cumulative importance weights at terminal state
            # (where reward is issued) for episodes of same length
            ratios = horizon_df.ratio.values.reshape(-1, horizon)
            cum_ratios = np.prod(ratios, axis=1)
            weights = cum_ratios / np.sum(cum_ratios)

            # compute same-horizon WIS at terminal state
            rewards = horizon_df.reward.values[horizon-1::horizon]
            wis.append(np.sum(weights * rewards))

            # horizon weight proportional to support
            wh.append(weights.shape[0])

        # weigh WIS by horizon weights
        norm_wh = np.array(wh) / np.sum(wh)
        return np.array(wis).dot(norm_wh)


if __name__ == '__main__':
    # behavior policy
    behavior_policy_file = 'physician_policy/amsterdam-umc-db_aggregated_full_cohort_1h_knn/valid_behavior_policy.csv'
    behavior_df = pd.read_csv(behavior_policy_file)
    behavior_policy = behavior_df.filter(regex='\d+').values  # -> 25 actions marked by integers 0 to 24!

    # random policy
    random_policy = np.random.uniform(0, 1, behavior_policy.shape)  # Noise
    random_policy = random_policy / np.sum(random_policy, axis=1, keepdims=True)

    estimator = PHWIS(behavior_policy_file)
    print('PHWIS:')
    print('Behavior policy:', estimator(behavior_policy))
    print('Random policy:  ', estimator(random_policy))

    # Sanity unbiased w.r.t. behavior policy?
    rewards = behavior_df.reward.values
    exp_reward = np.mean(rewards[rewards != 0])
    print('\nTrue reward behavior policy:', exp_reward)


    
    
