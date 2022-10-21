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
            return torch.sum(action_probs * q_pred, axis=1)
        return q_pred


class FittedQEvaluation:
    """ Implementation of the Bootstrapped Fitted Q-Evaluation (FQE) estimator for
        Off-policy Policy Evaluation (OPE). For details, see:
        http://proceedings.mlr.press/v139/hao21b/hao21b.pdf
    """
    def __init__(self, state_dims, num_actions, num_timesteps, gamma=0.9, lrate=1e-2, initial_iters=1000, step_iters=500, reg=1e-2):
        self._num_timesteps = num_timesteps
        self._gamma = gamma
        self._lrate = lrate

        # Model + warm start parameters
        self._estimator = FuncApproximator(state_dims + 1, num_actions)
        self._initial_iters = initial_iters
        self._step_iters = step_iters

        self._criterion = torch.nn.MSELoss()
        self._optimizer = torch.optim.SGD(self._estimator.parameters(), lr=lrate, weight_decay=reg)
        self.__fitted = False

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

        # Add timestep feature to states and next_states
        num_episodes = states.shape[0] // (self._num_timesteps - 1)
        timesteps = torch.arange(self._num_timesteps - 1).repeat(num_episodes).unsqueeze(1)
        states = torch.concat([states, timesteps], dim=1)
        next_states = torch.concat([next_states, timesteps + 1], dim=1)

        # Warm start -- continue training from last estimator
        iters = self._step_iters if self.__fitted else self._initial_iters

        # Mask out expected future reward at next state if terminal
        reward_mask = (rewards == 0).float()

        # Perform policy iteration
        pbar = tqdm(range(iters))
        for _ in pbar:

            # Bootstrapped target using policy πe!
            with torch.no_grad():
                q_next = rewards + self._gamma * reward_mask * self._estimator(next_states, action_probs=policy_next_action_probs)

            # Current Q-estimate
            q_pred = self._estimator(states, hard_actions=actions)

            self._optimizer.zero_grad()
            self._criterion(q_pred, q_next).backward()
            self._optimizer.step()

            # Show Q- and V-estimates during training
            start_q_pred = q_pred[0::self._num_timesteps - 1]  # Compute only over start-states!
            pbar.set_postfix({'avg_q': torch.mean(q_pred).item(),
                              'max_q': torch.max(q_pred).item(),
                              'avg_v_start': torch.mean(start_q_pred).item()})

        # Indicate that estimator has some fit
        self.__fitted = True
        return self

    def value(self, start_states, policy_action_probs):
        """
        Returns the estimated state value, V(s), according to the
        greedy policy on which the FQE instance was fitted.
        """
        if not self.__fitted:
            raise Exception('Estimator has not been fit; Call fit().')
        # Add timestep column to input (0 as we are concerned with start states)
        start_states = np.column_stack([start_states, np.zeros(start_states.shape[0])])

        # Average state value over start states
        start_states = torch.Tensor(start_states)
        action_probs = torch.Tensor(policy_action_probs)
        state_values = self._estimator(start_states, action_probs=action_probs)
        return torch.mean(state_values).detach().numpy()


if __name__ == '__main__':
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv')

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

    #####################
    #     Policies
    #####################

    # Behavior policy
    behavior_df = pd.read_csv('physician_policy/roggeveen_4h/mimic-iii_train_behavior_policy.csv')
    behavior_action_probs = behavior_df[[str(i) for i in range(25)]].values  # assume 25 actions
    behavior_next_action_probs = np.delete(behavior_action_probs, first_state_ids, axis=0)

    # Zero policy
    zerodrug_action_probs = np.zeros(behavior_action_probs.shape)
    zerodrug_action_probs[:, 0] = 1
    zerodrug_next_action_probs = np.delete(zerodrug_action_probs, first_state_ids, axis=0)

    # Random policy
    np.random.seed(42)
    random_action_probs = np.random.uniform(0, 1, behavior_action_probs.shape)
    random_action_probs = random_action_probs / np.sum(random_action_probs, axis=1, keepdims=True)
    random_next_action_probs = np.delete(random_action_probs, first_state_ids, axis=0)

    #####################
    #     Estimation
    #####################

    # Fit FQEs
    print('V(s_0):')
    fqe = lambda: FittedQEvaluation(state_dims=states.shape[1], num_actions=25, num_timesteps=18)

    estimator = fqe().fit(states, actions, rewards, next_states, policy_next_action_probs=behavior_next_action_probs)
    print('Behavior: ', estimator.value(all_states[first_state_ids], behavior_action_probs[first_state_ids]))

    estimator = fqe().fit(states, actions, rewards, next_states, policy_next_action_probs=zerodrug_next_action_probs)
    print('Zero-drug:', estimator.value(all_states[first_state_ids], zerodrug_action_probs[first_state_ids]))

    estimator = fqe().fit(states, actions, rewards, next_states, policy_next_action_probs=random_next_action_probs)
    print('Random:   ', estimator.value(all_states[first_state_ids], random_action_probs[first_state_ids]))

    # Sanity check (note: FQE is biased but relative estimates should make sense)
    rewards = all_rewards.reshape(-1, 18)[:, :-1]
    gamma_t = np.power(0.9, np.arange(17))[np.newaxis]
    total_reward = np.mean(np.sum(gamma_t * rewards, axis=1))
    print('\nEmpirical discounted reward of behavior policy:', total_reward)



