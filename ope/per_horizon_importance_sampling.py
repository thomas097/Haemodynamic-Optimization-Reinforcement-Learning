import numpy as np
import pandas as pd


class PHWIS:
    def __init__(self, behavior_policy_file, gamma=1.0):
        """ Implementation of Per-Horizon Weighted Importance Sampling (PHWIS) for Off-policy Evaluation (OPE)
        of treatment policies from log data with variable-length horizons.
        For details, please refer to https://www.ijcai.org/proceedings/2018/0729.pdf
        :param behavior_policy_file:  Path to DataFrame containing behavior policy (columns '0'-'24'),
                                      chosen actions ('action') and associated rewards ('reward')
        """
        # determine probability of chosen action under behavior policy
        behavior_df = pd.read_csv(behavior_policy_file)
        self._actions, action_probs = self._get_action_probs(behavior_df)
        self._ess = None

        # determine horizon of each episode
        horizon = behavior_df.groupby('episode').size()

        # count time steps in episode
        timesteps = behavior_df.groupby('episode').action.transform(lambda ep: np.arange(len(ep)))

        # combine into DataFrame
        self._df = pd.DataFrame({
            'episode': behavior_df.episode.astype(int),
            'action': behavior_df.action,
            'reward': behavior_df.reward,
            'timestep': timesteps,
            'gamma': gamma ** timesteps,
            'horizon': behavior_df.episode.apply(lambda ep: horizon.loc[ep]),
            'probs_b': action_probs
        })

    def _get_action_probs(self, df):
        """ Extract chosen actions and associated probabilities of behavior policy
        """
        actions = df.action.values[:, np.newaxis].astype(int)
        action_probs = df.filter(regex='\d+').values
        chosen_action_probs = np.take_along_axis(action_probs, indices=actions, axis=1).flatten()
        return actions, chosen_action_probs

    @property
    def ess(self):
        """ Compute effective sample size (ESS) as 1 / sum(w_i^2) and
        weigh episodes by the contributions of their horizons
        """
        # weigh importance weights by normalized horizon weight
        ess = np.array([1 / np.sum(w ** 2) for w in self._ess])
        #supp = np.array([w.shape[0] for w in self._ess])
        return np.sum(ess)#.dot(supp / np.sum(supp))

    def ratios(self, pi_e, episodes=None):
        """ Returns a DataFrame of cumulative importance ratios ('rho') for each timestep and episode
        :param pi_e:      Table of action probs under πe with shape (num_states, num_actions)
        :param episodes:  Subset of episodes to consider in estimating V^πe
        :returns:         DataFrame with importance ratios and horizon weights
        """
        # compute importance ratio of action prob for chosen action under pi_e and pi_b
        df = self._df.copy()
        df['probs_e'] = np.take_along_axis(pi_e, indices=self._actions, axis=1).flatten()
        df['ratio'] = df.probs_e / df.probs_b
        df['ratio'] = np.clip(df.ratio ** 0.25, a_min=0.5, a_max=2) # non-lin OPE

        # filter episodes (optional))
        if episodes is not None:
            df = df[df.episode.isin(episodes)]

        # compute importance weight at each timestep and normalize over episodes of the same length
        df['rho'] = df.groupby('episode').ratio.cumprod()
        df['norm_rho'] = df.groupby(['horizon', 'timestep']).rho.apply(lambda rho: self._normalize(rho))

        # track weights at terminal state to compute sample size
        self._ess = [w.norm_rho.values[h - 1::h] for h, w in df.groupby('horizon')]

        # compute contribution of horizon wh as '#episodes with horizon' / '#episodes'
        n_episodes = df.episode.nunique()
        df['wh'] = df.groupby('horizon').episode.transform(lambda x: x.nunique()) / n_episodes
        return df

    def _normalize(self, ratios):
        """ Normalizes ratios """
        total = np.sum(ratios)
        if total == 0:
            return np.full(shape=ratios.shape, fill_value=1 / ratios.shape[0])
        return ratios / total

    def __call__(self, pi_e, episodes=None):
        """ Computes the PHWIS estimate of V^πe
        :param pi_e:      Table of action probs acc. to πe with shape (num_states, num_actions)
        :param episodes:  Subset of episodes to consider in estimating V^πe
        :returns:         Estimate of mean V^πe
        """
        df = self.ratios(pi_e, episodes=episodes)
        return (df.wh * df.gamma * df.norm_rho * df.reward).sum()


if __name__ == '__main__':
    ## Step 0: Policies

    # behavior policy
    behavior_policy_file = 'physician_policy/amsterdam-umc-db_v3_aggregated_full_cohort_2h_mlp/valid_behavior_policy.csv'
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

    ## Step 1: Estimate performance

    estimator = PHWIS(behavior_policy_file, gamma=0.95)
    print('PHWIS:')
    behavior_phwis = estimator(behavior_policy)
    print('Behavior policy: %.3f (%.2f)' % (behavior_phwis, estimator.ess))

    semi_random_phwis = estimator(semi_random_policy)
    print('Semi-random policy: %.3f (%.2f)' % (semi_random_phwis, estimator.ess))

    random_phwis = estimator(random_policy)
    print('Random policy: %.3f (%.2f)' % (random_phwis, estimator.ess))

    # Sanity check: what return do we expect from the behavior policy?
    behavior_policy = pd.read_csv(behavior_policy_file)
    behavior_policy['gamma'] = behavior_policy.groupby('episode').reward.transform(lambda x: 0.95 ** np.arange(len(x)))
    behavior_policy['discounted_reward'] = behavior_policy.gamma * behavior_policy.reward
    expected_return = behavior_policy.groupby('episode').discounted_reward.sum().mean()
    print('\nExpected return:', expected_return)


    
    
