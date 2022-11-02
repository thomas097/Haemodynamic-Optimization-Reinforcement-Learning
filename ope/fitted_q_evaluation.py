import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


class FuncApproximator(torch.nn.Module):
    def __init__(self, state_dims, num_actions):
        super().__init__()
        self._estimator = torch.nn.Sequential(
            torch.nn.Linear(state_dims, num_actions)
        )

    def forward(self, states, hard_actions=None, action_probs=None):
        q_pred = self._estimator(states)
        if hard_actions is not None:
            return q_pred.gather(dim=1, index=hard_actions.unsqueeze(1))[:, 0]
        if action_probs is not None:
            return torch.sum(action_probs * q_pred, axis=1)  # Expected SARSA
        return q_pred


class FQEDataset:
    """ Convenience wrapper around dataset
    """
    def __init__(self):
        self.state_dims = 0
        self.num_timesteps = 0
        self.all_states = None
        self.all_actions = None
        self.states = None
        self.actions = None
        self.rewards = None
        self.next_states = None
        self.last_state_idx = []
        self.first_state_idx = []


class FittedQEvaluation:
    """ Implementation of the Fitted Q-Evaluation (FQE) estimator for Off-policy
        Policy Evaluation (OPE). For details, see:
        http://proceedings.mlr.press/v139/hao21b/hao21b.pdf
    """
    def __init__(self, training_file, num_actions=25, gamma=0.9, lrate=1e-2, iters=1000, reg=1e-2):
        # Unpack training data file as FQEDataset object holding states, actions, rewards, etc.
        self._train = self._unpack_training_file(training_file)

        # Model + hyperparameters
        self._estimator = FuncApproximator(self._train.state_dims, num_actions)
        self._gamma = gamma
        self._iters = iters

        # Optimization
        self._criterion = torch.nn.MSELoss()
        self._optimizer = torch.optim.SGD(self._estimator.parameters(), lr=lrate, weight_decay=reg)
        self._fitted = False

    @property
    def fitted(self):
        return self._fitted

    @staticmethod
    def _unpack_training_file(training_file):
        # Unpack training_file file into states, actions, etc.
        train_df = pd.read_csv(training_file).reset_index(drop=True)
        all_states = train_df.filter(regex='x\d+').values
        all_actions = train_df['action'].values
        all_rewards = train_df['reward'].values

        # Determine first and last states
        first_state_idx = [g.index[0] for _, g in train_df.groupby('episode')]
        last_state_idx = [g.index[-1] for _, g in train_df.groupby('episode')]
        num_timesteps = np.max(train_df.groupby('episode').action.size())

        # Extract states, actions, rewards and 'next states'
        states = np.delete(all_states, last_state_idx, axis=0)
        actions = np.delete(all_actions, last_state_idx, axis=0)
        rewards = np.delete(all_rewards, first_state_idx, axis=0)
        next_states = np.delete(all_states, first_state_idx, axis=0)

        # Pack data into FQEDataset object
        dataset = FQEDataset()
        dataset.state_dims = states.shape[1]
        dataset.num_timesteps = num_timesteps
        dataset.all_states = torch.Tensor(all_states)
        dataset.all_actions = torch.LongTensor(all_actions)
        dataset.states = torch.Tensor(states)
        dataset.actions = torch.LongTensor(actions)
        dataset.rewards = torch.Tensor(rewards)
        dataset.next_states = torch.Tensor(next_states)
        dataset.last_state_idx = last_state_idx
        dataset.first_state_idx = first_state_idx
        return dataset

    def fit(self, policy_action_probs):
        """
        Fits FQE estimator for Q^pi to states, actions, rewards and next
        states obtained through a behavior policy.
        """
        # Limit policy's action probabilities to pi(*|s')
        policy_next_action_probs = np.delete(policy_action_probs, self._train.first_state_idx, axis=0)
        policy_next_action_probs = torch.Tensor(policy_next_action_probs)

        # Mask out expected future reward at next state if terminal
        reward_mask = (self._train.rewards == 0).float()

        # Perform policy iteration
        with tqdm(range(self._iters)) as pbar:
            for _ in pbar:
                # Q-estimate
                q_pred = self._estimator(self._train.states, hard_actions=self._train.actions)

                # Bootstrapped target using policy πe!
                with torch.no_grad():
                    exp_future_reward = self._estimator(self._train.next_states, action_probs=policy_next_action_probs)
                    q_next = self._train.rewards + self._gamma * reward_mask * exp_future_reward

                # Update!
                self._optimizer.zero_grad()
                self._criterion(q_pred, q_next).backward()
                self._optimizer.step()

                pbar.set_postfix({'avg_q': torch.mean(q_pred).item()})

        self._fitted = True
        return self

    def state_value(self, policy_action_probs):
        """
        Returns the estimated state value, V(s), according to the
        evaluation policy on which the FQE instance was fitted.
        """
        if not self._fitted:
            raise Exception('Estimator has not been fitted; Call fit().')

        actions_probs = torch.Tensor(policy_action_probs)
        return self._estimator(self._train.all_states, action_probs=actions_probs).detach().numpy()

    def state_action_value(self):
        """
        Returns the estimated state-action value, Q(s, a), according to the
        evaluation policy on which the FQE instance was fitted.
        """
        if not self._fitted:
            raise Exception('Estimator has not been fitted; Call fit().')

        return self._estimator(self._train.all_states).detach().numpy()


if __name__ == '__main__':
    # Behavior policy
    behavior_df = pd.read_csv('physician_policy/roggeveen_4h/mimic-iii_train_behavior_policy.csv')
    behavior_action_probs = behavior_df.filter(regex='\d+').values  # assume 25 actions

    # Zero policy
    zerodrug_action_probs = np.zeros(behavior_action_probs.shape)
    zerodrug_action_probs[:, 0] = 1

    # Random policy
    np.random.seed(42)
    random_action_probs = np.random.uniform(0, 1, behavior_action_probs.shape)
    random_action_probs = random_action_probs / np.sum(random_action_probs, axis=1, keepdims=True)

    # Fit FQEs
    training_file = '../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv'
    behavior_estimator_fqe = FittedQEvaluation(training_file).fit(behavior_action_probs)
    zerodrug_estimator_fqe = FittedQEvaluation(training_file).fit(zerodrug_action_probs)
    random_estimator_fqe = FittedQEvaluation(training_file).fit(random_action_probs)

    print('FQE - V(s0):')
    print('Behavior: ', np.mean(behavior_estimator_fqe.state_value(behavior_action_probs)))
    print('Zero-drug:', np.mean(zerodrug_estimator_fqe.state_value(zerodrug_action_probs)))
    print('Random:   ', np.mean(random_estimator_fqe.state_value(random_action_probs)))


