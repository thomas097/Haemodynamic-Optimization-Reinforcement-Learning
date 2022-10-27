import numpy as np
import pandas as pd
from sklearn.preprocessing import label_binarize


class SoftOPC:
    """ Implementation of the Soft Off-Policy Classification (SoftOPC) method
        for OPE (Irpan et al., 2019)
        For details, see: https://openreview.net/pdf?id=SJlH4fJaiE
    """
    def __init__(self, behavior_policy_file, compute_prior=False, drop_terminal_states=True):
        phys_df = pd.read_csv(behavior_policy_file)
        if drop_terminal_states:
            phys_df = phys_df[phys_df.reward.notna()]

        # Actions taken by behavior policy (i.e. physician)
        self._actions = np.expand_dims(phys_df['action'], axis=1).astype(int)

        # Masks to indicate trajectories ending in success (positive final reward)
        is_success = phys_df.groupby('episode', sort=False).reward.transform(self._ends_in_success)
        self._is_success = is_success.values.astype(np.bool)

        # Positive class prior P(Y=1)
        self._prior = np.mean(self._is_success) if compute_prior else 1.0

    @staticmethod
    def _ends_in_success(episode):
        return np.ones(len(episode)) if episode.max() > 0 else np.zeros(len(episode))

    def __call__(self, qvals):
        qvals_all = np.take_along_axis(qvals, self._actions, axis=1).flatten()
        qvals_success = qvals_all[self._is_success]
        return self._prior * np.mean(qvals_success) - np.mean(qvals_all)


if __name__ == '__main__':
    softopc = SoftOPC('physician_policy/roggeveen_4h_with_cv/mimic-iii_valid_behavior_policy.csv')

    # Success policy (agent assigning high Q-values to actions only when outcome is success)
    def ends_in_positive_reward(episode):
        return np.ones(len(episode)) if episode.max() > 0 else np.zeros(len(episode))

    df = pd.read_csv('physician_policy/roggeveen_4h_with_cv/mimic-iii_valid_behavior_policy.csv')
    success_states = df[df.reward.notna()].groupby('episode', sort=False).reward.transform(ends_in_positive_reward)
    success_qvals = np.expand_dims(success_states, axis=1).repeat(25, axis=1)

    # Random policy
    random_qvals = np.random.uniform(-1, 1, (25551, 25))

    # Zero-drug policy
    zerodrug_qvals = np.zeros((25551, 25))
    zerodrug_qvals[:, 0] = 1

    print('SoftOPC:')
    print('Succesful policy:', softopc(success_qvals))
    print('Random policy:   ', softopc(random_qvals))
    print('Zero-drug policy:', softopc(zerodrug_qvals))


