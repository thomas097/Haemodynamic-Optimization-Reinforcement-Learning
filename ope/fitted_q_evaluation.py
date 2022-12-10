import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


class MLP(torch.nn.Module):
    def __init__(self, state_dims, num_actions, hidden_dims=24):
        """ Simple Multi-Layer Perceptron for estimating the FQE Q-function
        :param state_dims:  Number of state space features
        :param num_actions: Number of actions
        :param hidden_dims: Number of hidden units of feature networks
        """
        super(MLP, self).__init__()
        # feature network + Q-network
        self._estimator = torch.nn.Sequential(
            torch.nn.Linear(state_dims, hidden_dims),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dims, hidden_dims),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(hidden_dims, num_actions)
        )

    def forward(self, states, hard_actions=None, action_probs=None):
        """ Returns q values of actions
        :param states:        FloatTensor of shape (num_states, num_features)
        :param hard_actions:  Actions chosen by physician (optional). If given, will
                              only return Q-estimates of behavior policy's actions
        :param action_probs:  Probability of action under evaluation policy (optional).
                              If given, will return expected return using SARSA target
        """
        q_pred = self._estimator(states)
        if hard_actions is not None:
            return q_pred.gather(dim=1, index=hard_actions.unsqueeze(1))[:, 0]
        if action_probs is not None:
            return torch.sum(action_probs * q_pred, axis=1)  # Expected SARSA
        return q_pred


class FQEDataset:
    def __init__(self, training_file):
        """ Convenience wrapper to parse and store training dataset
        """
        self.state_dims = 0
        self.all_states = None
        self.all_actions = None
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.last_state_idx = []
        self.first_state_idx = []
        self._unpack_(training_file)

    def _unpack_(self, training_file):
        # Unpack training_file file into states, actions, etc.
        df = pd.read_csv(training_file).reset_index(drop=True)
        all_states = df.filter(regex='x\d+').values
        all_actions = df.action.values
        all_rewards = df.reward.values

        # Determine first and last states
        first_state_idx = [g.index[0] for _, g in df.groupby('episode')]
        last_state_idx = [g.index[-1] for _, g in df.groupby('episode')]

        # Extract states, actions, rewards and 'next states'
        states = np.delete(all_states, last_state_idx, axis=0)
        actions = np.delete(all_actions, last_state_idx, axis=0)
        rewards = np.delete(all_rewards, first_state_idx, axis=0)
        next_states = np.delete(all_states, first_state_idx, axis=0)

        # Pack data into FQEDataset object
        self.episodes = df.episode
        self.state_dims = states.shape[1]
        self.all_states = torch.Tensor(all_states)
        self.all_actions = torch.LongTensor(all_actions)
        self.last_state_idx = last_state_idx
        self.first_state_idx = first_state_idx
        self.states = torch.Tensor(states)
        self.actions = torch.LongTensor(actions)
        self.rewards = torch.Tensor(rewards)
        self.next_states = torch.Tensor(next_states)


class FittedQEvaluation:
    def __init__(self, training_file, num_actions=25, gamma=0.9, lrate=1e-1, iters=1000, reward_range=(-100, 100)):
        """ Implementation of Fitted Q-Evaluation (FQE) for Off-policy Policy Evaluation (OPE)
        For details, see: http://proceedings.mlr.press/v139/hao21b/hao21b.pdf
        :param training_file:   Dataset used to train FQE estimator (only aggregated dataset is supported for now)
        :param num_actions:     Number of possible actions by agent
        :param gamma:           Discount factor to trade-off immediate against future reward
        :param lrate:           Learning rate
        :param iters:           Training iterations
        :param reward_range:    (min, max) range of rewards
        """
        # Pack training data file into a FQEDataset object holding states, actions, rewards, etc.
        self._train = FQEDataset(training_file)
        self._estimator = MLP(self._train.state_dims, num_actions)
        self._min_reward, self._max_reward = reward_range
        self._gamma = gamma
        self._iters = iters
        self._fitted = False

        self._criterion = torch.nn.MSELoss()
        self._optimizer = torch.optim.SGD(self._estimator.parameters(), lr=lrate)

    @property
    def fitted(self):
        return self._fitted

    def fit(self, pi_e):
        """ Fits FQE estimator for Q^pi_e with state-action-reward-nextstate transitions
        collected under the behavior policy
        :param pi_e:  Tensor of action probabilities for each state
        """
        # limit policy's action probabilities to pi(*|s')
        policy_next_action_probs = np.delete(pi_e, self._train.first_state_idx, axis=0)
        policy_next_action_probs = torch.Tensor(policy_next_action_probs)

        # sanity check: same number of states?
        assert policy_next_action_probs.shape[0] == self._train.next_states.shape[0]

        # define mask to mask out expected reward at next state in terminal states
        reward_mask = (self._train.rewards == 0).float()

        # perform policy iteration
        avg_q_vals = []
        with tqdm(range(self._iters)) as pbar:

            for _ in pbar:
                # bootstrapped targets
                with torch.no_grad():
                    exp_future_reward = self._estimator(self._train.next_states, action_probs=policy_next_action_probs)
                    q_next = self._train.rewards + self._gamma * reward_mask * exp_future_reward
                    q_next = torch.clamp(q_next, min=self._min_reward, max=self._max_reward)

                # Q-estimate
                q_pred = self._estimator(self._train.states, hard_actions=self._train.actions)

                # update!
                self._optimizer.zero_grad()
                self._criterion(q_pred, q_next).backward()
                self._optimizer.step()

                # print average Q-value to screen
                pbar.set_postfix({'avg_q': torch.mean(q_pred).item()})

        self._fitted = True
        return self

    def _limit_episodes(self, arr, episodes=None):
        """ Convenience function to limit episodes in arr to just a subset """
        if episodes is not None:
            return arr[self._train.episodes.isin(episodes)]
        else:
            return arr

    def state_value(self, pi_e, episodes=None):
        """ Returns the estimated state value, V(s), according to the
        evaluation policy on which the FQE instance was fitted
        :param pi_e:     Tensor of action probabilities for each state
        param episodes:  Episodes to restrict state-values to
        """
        if not self._fitted:
            raise Exception('Estimator has not been fitted; Call fit().')

        # limit states to specific episodes (if specified)
        states = self._limit_episodes(self._train.all_states, episodes=episodes)
        probs = self._limit_episodes(torch.Tensor(pi_e), episodes=episodes)

        return self._estimator(states, action_probs=probs).detach().numpy()

    def state_action_value(self, episodes=None):
        """ Returns the estimated state-action value, Q(s, a), according to the
        evaluation policy on which the FQE instance was fitted
        """
        if not self._fitted:
            raise Exception('Estimator has not been fitted; Call fit().')

        # limit states to specific episodes (if specified)
        states = self._limit_episodes(self._train.all_states, episodes=episodes)

        return self._estimator(states).detach().numpy()


if __name__ == '__main__':
    # Behavior policy
    behavior_df = pd.read_csv('physician_policy/amsterdam-umc-db_v3_aggregated_full_cohort_2h_mlp/valid_behavior_policy.csv')
    behavior_action_probs = behavior_df.filter(regex='\d+').values  # assume 25 actions

    # Zero policy
    zerodrug_action_probs = np.zeros(behavior_action_probs.shape)
    zerodrug_action_probs[:, 0] = 1

    # Random policy
    np.random.seed(42)
    random_action_probs = np.random.uniform(0, 1, behavior_action_probs.shape)
    random_action_probs = random_action_probs / np.sum(random_action_probs, axis=1, keepdims=True)

    # Fit FQEs
    # Note: always use aggregated data file for training (time points in non-aggregated file will match!)
    training_file = '../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/valid.csv'
    behavior_estimator_fqe = FittedQEvaluation(training_file, gamma=0.95).fit(behavior_action_probs)
    print('Behavior: ', np.mean(behavior_estimator_fqe.state_value(behavior_action_probs)))

    zerodrug_estimator_fqe = FittedQEvaluation(training_file, gamma=0.95).fit(zerodrug_action_probs)
    print('Zero-drug:', np.mean(zerodrug_estimator_fqe.state_value(zerodrug_action_probs)))

    random_estimator_fqe = FittedQEvaluation(training_file, gamma=0.95).fit(random_action_probs)
    print('Random:   ', np.mean(random_estimator_fqe.state_value(random_action_probs)))


