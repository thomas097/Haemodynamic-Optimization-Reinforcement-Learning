import torch
import numpy as np
import pandas as pd
from tqdm import tqdm


class FuncApproximator(torch.nn.Module):
    def __init__(self, state_dims, num_actions, d_hidden=24):
        super().__init__()
        self._estimator = torch.nn.Sequential(
            torch.nn.Linear(state_dims, num_actions),
            # torch.nn.LeakyReLU(),
            # torch.nn.Linear(d_hidden, num_actions),
        )

    def forward(self, states, hard_actions=None, action_probs=None):
        q_pred = self._estimator(states)
        if hard_actions is not None:
            return q_pred.gather(dim=1, index=hard_actions.unsqueeze(1))[:, 0]
        if action_probs is not None:
            return torch.sum(action_probs * q_pred, axis=1)
        return q_pred


class FittedQEvaluation:
    """ Implementation of the Bootstrapped Fitted Q-Evaluation (FQE) estimator for
        Off-policy Policy Evaluation (OPE). For details, see:
        http://proceedings.mlr.press/v139/hao21b/hao21b.pdf
    """
    def __init__(self, state_dims, num_actions, gamma=0.9, lrate=1e-2, initial_iters=1000, step_iters=100, l2=1e-3):
        self._estimator = FuncApproximator(state_dims, num_actions)
        self._gamma = gamma
        self._lrate = lrate

        # Warm start parameters
        self._fitted = False
        self._initial_iters = initial_iters
        self._step_iters = step_iters

        # Define MSE loss and optimizer
        self._criterion = torch.nn.MSELoss()
        self._optimizer = torch.optim.SGD(self._estimator.parameters(), lr=self._lrate, weight_decay=l2)

    def fit(self, states, actions, rewards, next_states, policy_next_action_probs):
        """
        Fits FQE estimator for Q^pi to states, actions, rewards
        and next states obtained through a behavior policy.
        """
        # To Pytorch Tensors
        states = torch.Tensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.Tensor(rewards)
        next_states = torch.Tensor(next_states)
        policy_next_action_probs = torch.Tensor(policy_next_action_probs)

        # Warm start -- continue training from last estimator
        iters = self._step_iters if self._fitted else self._initial_iters

        # Mask out expected future reward at next state if terminal
        reward_mask = (rewards == 0).float()

        # Perform policy iteration
        pbar = tqdm(range(iters))
        for _ in pbar:

            # Bootstrapped target using greedy policy πe!
            with torch.no_grad():
                q_next = rewards + self._gamma * reward_mask * self._estimator(next_states, action_probs=policy_next_action_probs)

            # What's the estimate now?
            q_pred = self._estimator(states, hard_actions=actions)
            pbar.set_postfix({'Avg. Q': torch.mean(q_pred).item()})

            self._optimizer.zero_grad()
            self._criterion(q_pred, q_next).backward()
            self._optimizer.step()

        # Mark that estimator has some fit
        self._fitted = True

    def value(self, states, policy_actions):
        """
        Returns the estimated state value, V(s), according to the
        greedy policy on which the FQE instance was fitted.
        """
        states = torch.Tensor(states)
        actions = torch.LongTensor(policy_actions).unsqueeze(1)
        state_values = self._estimator(states).gather(dim=1, index=actions)
        return torch.mean(state_values).detach().numpy()


if __name__ == '__main__':
    train_df = pd.read_csv('../../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv')

    # Unpack training dataset into states, actions, rewards and episode IDs
    meta_data = ['icustay_id', 'timestep', 'max_vp_shifted', 'total_iv_fluid_shifted', 'reward', 'action', 'state_sirs']
    all_states = train_df[[c for c in train_df.columns if c not in meta_data]].values.astype(np.float32)
    all_actions = train_df['action'].values.astype(np.uint8)
    all_rewards = train_df['reward'].values.astype(np.float32)
    episodes = train_df['icustay_id'].values.astype(np.uint32)

    # Define (non-terminal) states as not 'last states' in trajectory
    last_state_ids = len(episodes) - np.unique(episodes[::-1], return_index=1)[1] - 1
    states = np.delete(all_states, last_state_ids, axis=0)
    actions = np.delete(all_actions, last_state_ids, axis=0)
    rewards = np.delete(all_rewards, last_state_ids, axis=0)

    # Define 'next states' as not 'first states'
    first_state_ids = np.unique(episodes, return_index=1)[1]
    next_states = np.delete(all_states, first_state_ids, axis=0)

    # Behavior policy
    behavior_df = pd.read_csv('roggeveen_4h/mimic-iii_train_behavior_policy.csv')
    behavior_action_probs = behavior_df[[str(i) for i in range(25)]].values  # assume 25 actions
    behavior_next_action_probs = np.delete(behavior_action_probs, first_state_ids, axis=0)

    # Random policy
    random_next_action_probs = np.random.uniform(0, 1, behavior_next_action_probs.shape)
    random_next_action_probs = random_next_action_probs / np.sum(random_next_action_probs, axis=1, keepdims=True)

    # Fit FQE
    fqe = FittedQEvaluation(state_dims=states.shape[1], num_actions=25)
    fqe.fit(states, actions, rewards, next_states, policy_next_action_probs=random_next_action_probs)

    # Evaluate
    print('V(s):', fqe.value(all_states[first_state_ids], all_actions[first_state_ids]))

    # Sanity check: Empirical reward under behavior policy
    rewards = all_rewards.reshape(-1, 18)[:, :-1]
    gamma_t = np.power(0.9, np.arange(17))[np.newaxis]
    total_reward = np.mean(np.sum(gamma_t * rewards, axis=1))
    print('\nEmpirical discounted reward of behavior policy:', total_reward)



