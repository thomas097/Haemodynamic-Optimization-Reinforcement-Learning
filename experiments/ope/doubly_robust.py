import numpy as np
import pandas as pd
from fitted_q_iteration import FittedQIteration


class WeightedDoublyRobust:
    def __init__(self, behavior_policy_file, states, actions, rewards, episodes, max_iters=10, gamma=0.9, verbose=False):
        """ Implementation of the Weighted Doubly Robust (WDR) estimator for Off-policy
            Policy Evaluation (OPE). We use Weighted Importance Sampling estimator for the IS
            part and a Fitted Q-Iteration estimator using Random Forests (RF) for the DM part.
            Please see https://arxiv.org/pdf/1802.03493.pdf for details.

            Params
            behavior_policy_file: DataFrame containing action probabilities (columns '0'-'24') for behavior policy,
                                  chosen actions ('action') and associated rewards ('reward').
            gamma:                Discount factor
        """
        # pi_b -> estimated action distribution of physician
        behavior_df = pd.read_csv(behavior_policy_file)
        self._pi_b = self._get_action_probabilities(behavior_df)

        # Action indices (e.g. int(5) -> action nr. 6)
        self._action_idx = np.expand_dims(behavior_df['action'], axis=1).astype(int)

        # Infer max number of time steps T
        self._timesteps = behavior_df.groupby('episode').size().max()

        # Table of non-discounted rewards (incl. NaNs at terminal state!)
        self._rewards = self._to_table(behavior_df['reward'].values)
        self._gamma = gamma

        # Fit DM estimator (FQI) yielding estimates for Q(s, a) and V(s)
        self._estimator, self._Q, self._V = self._fit_DM(states, actions, rewards, episodes, max_iters, verbose)

    def _fit_DM(self, states, actions, rewards, episodes, max_iters, verbose):
        # Fit FQI model to (state, action, reward) tuples
        estimator = FittedQIteration(states, actions, rewards, episodes, method='rf', max_iters=max_iters,
                                     gamma=self._gamma, verbose=verbose)

        # Use estimator to obtain state- and state-value estimates (V and Q)
        q_value = self._to_table(estimator.Q(states, actions))
        v_value = self._to_table(estimator.V(states))
        return estimator, q_value, v_value

    def _to_table(self, flat_list):
        # Converts a flat list of concatenated episodes to a table of shape (num_episodes, num_time_steps)
        return flat_list.reshape(-1, self._timesteps)

    @staticmethod
    def _get_action_probabilities(behavior_policy_df):
        # Return numeric columns (assumed to belong to actions, i.e., '0' - '24')
        cols = sorted([c for c in behavior_policy_df.columns if c.isnumeric()], key=lambda c: int(c))
        return behavior_policy_df[cols].values.astype(np.float64)

    def __call__(self, pi_e):
        """ Computes the WIS estimate of V^πe.

            pi_e:       Table of action probs acc. to πe with shape (num_states, num_actions)

            Returns:    Estimate of mean V^πe
        """
        # Extract table with action probabilities acc. to πe (πb) w.r.t. chosen actions
        action_probs_e = self._to_table(np.take_along_axis(pi_e, self._action_idx, axis=1))
        action_probs_b = self._to_table(np.take_along_axis(self._pi_b, self._action_idx, axis=1))

        # Drop terminal states where no rewards are received
        action_probs_e = action_probs_e[:, :-1]
        action_probs_b = action_probs_b[:, :-1]
        rewards = self._rewards[:, :-1]
        q_estimate = self._Q[:, :-1]
        v_estimate = self._V[:, :-1]

        # Compute cumulative importance ratio
        ratio = np.cumprod(action_probs_e / action_probs_b, axis=1)
        w_t = ratio / ratio.sum(axis=0, keepdims=True)

        # Compute gamma^t for 0:T-1
        gamma_t = np.power(self._gamma, np.arange(self._timesteps - 1))[np.newaxis]

        # Shift w_{t-1} one step forward to time t
        w_t_minus_1 = np.zeros(w_t.shape)
        w_t_minus_1[:, 1:] = w_t[:, :-1]

        # Compute WIS and DM part
        wis = gamma_t * w_t * rewards
        dm = gamma_t * (w_t * q_estimate - w_t_minus_1 * v_estimate)
        return np.sum(wis) - np.sum(dm)


if __name__ == '__main__':
    training_data = pd.read_csv('../datasets/mimic-iii/roggeveen/mimic-iii_train.csv')

    # Unpack training dataset into states, actions, rewards and episode IDs
    meta_data = ['icustay_id', 'timestep', 'max_vp_shifted', 'total_iv_fluid_shifted', 'reward', 'action', 'state_sirs']
    actions = training_data['action'].values.astype(np.uint8)
    rewards = training_data['reward'].values
    episodes = training_data['icustay_id'].values.astype(np.uint64)
    states = training_data[[c for c in training_data.columns if c not in meta_data]].values

    # Behavior policy
    behavior_df = pd.read_csv('../physician_policy/mimic-iii_train_behavior_policy.csv')
    behavior_policy = behavior_df[[str(i) for i in range(25)]].values

    # Create WDR estimator
    wdr = WeightedDoublyRobust('../physician_policy/mimic-iii_train_behavior_policy.csv',
                               states, actions, rewards, episodes, verbose=True)

    print('WDR:')
    print('behavior:', wdr(behavior_policy))

    # Sanity check: Expected discounted reward of behavior policy?
    rewards = rewards.reshape(-1, 18)[:, :-1]
    gamma = np.power(0.9, np.arange(17))[np.newaxis]
    discounted_reward = np.mean(np.sum(gamma * rewards, axis=1))
    print('\nTrue reward of pi_b:', discounted_reward)
