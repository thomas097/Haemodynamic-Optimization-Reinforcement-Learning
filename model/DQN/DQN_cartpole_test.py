import gym
import numpy as np
import pandas as pd

from DQN import DuelingDQN, fit_dueling_double_DQN


class CartPoleTeacherModel:
    def __init__(self):
        # Yes... 4 parameters are enough to solve cartpole :)
        self._w = np.array([-0.04012991, 0.22891377, -0.1040997, 0.85112371])

    def sample(self, state):
        return 1 if state.dot(self._w) > 0 else 0


def cartpole(model=None, num_episodes=10, max_steps=500, epsilon=0.15, render=False, seed=None):
    """ Runs a model on the CartPole-v1 gym environment or, when model=None, generates
        off-policy samples to train model on. In case no model is provided, an epsilon
        greedy behavior policy is used based on a learned 4-parameter model.
        This function is meant as a benchmark for development of DQN.py for offline learning.

        Action space is 2 actions: left (0) and right (1)!
    """
    env = gym.make("CartPole-v1")
    if seed is not None:
        env.seed(seed)

    behavior_policy = CartPoleTeacherModel()

    episode, timestep, states, actions, rewards = [], [], [], [], []

    for ep in range(num_episodes):

        state = env.reset()
        for ts in range(max_steps):

            if render:
                env.render()

            if model is not None:
                action = model.sample(state[None])
            else:
                # Epsilon greedy behavior policy
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = behavior_policy.sample(state)

            next_state, reward, terminated, info = env.step(action)

            episode.append(ep)
            timestep.append(ts)
            states.append(state)
            actions.append(action)
            rewards.append(reward)

            if terminated:
                episode.append(ep)
                timestep.append(ts + 1)
                states.append(next_state)
                actions.append(np.NaN)  # no action in terminal state
                rewards.append(np.NaN)  # no reward in terminal state
                break

            state = next_state

    env.close()

    # Convert histories to DataFrame
    df = pd.DataFrame({'episode': episode,
                       'timestep': timestep,
                       'state_0': [s[0] for s in states],
                       'state_1': [s[1] for s in states],
                       'state_2': [s[2] for s in states],
                       'state_3': [s[3] for s in states],
                       'action': actions,
                       'reward': rewards}).reset_index(drop=True)
    return df


if __name__ == '__main__':
    np.random.seed(100)  # reproducibility

    # create DQN controller
    model = DuelingDQN(state_dim=4, num_actions=2, hidden_dims=(96,))

    # Sample dataset of N episodes for off-policy training
    dataset = cartpole(num_episodes=5000, seed=100)
    print(dataset)

    # Fit model to dataset
    fit_dueling_double_DQN(model=model,
                           dataset=dataset,
                           state_cols=['state_0', 'state_1', 'state_2', 'state_3'],
                           action_col='action',
                           reward_col='reward',
                           episode_col='episode',
                           timestep_col='timestep',
                           alpha=1e-4,
                           gamma=0.9,
                           tau=1e-2,
                           num_episodes=1000,
                           batch_size=32,
                           eval_func=lambda m: cartpole(m, num_episodes=100, seed=3),  # Evaluate on cartpole simulator!
                           eval_after=100,
                           scheduler_gamma=0.95,
                           step_scheduler_after=200)

    # Try out model in simulator
    cartpole(model=model, num_episodes=100, render=True)
