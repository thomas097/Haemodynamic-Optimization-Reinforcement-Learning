import gym
import numpy as np
import pandas as pd


def cartpole(model=None, num_episodes=100, max_timesteps=1000):
    """ Runs a model on the CartPole-v1 gym environment or generates
        off-policy samples to train model on (when model=None). In case
        no model is provided, a uniform random behavior policy is used.
        This function is primarily meant as a benchmark for development.
    """
    env = gym.make("CartPole-v1")

    histories = []

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode, timestep, states, actions, rewards = [], [], [], [], []

        for ts in range(max_timesteps):
            if model is not None:
                action = model.sample(state)
            else:
                action = env.action_space.sample()

            next_state, reward, terminated, info = env.step(action)
            total_reward += reward

            episode.append(ep)
            timestep.append(ts)
            states.append(state)
            actions.append(action)

            if terminated:
                rewards.append(-100)  # Negative reward is never given?
            else:
                rewards.append(reward)

            if terminated:
                episode.append(ep)
                timestep.append(ts + 1)
                states.append(next_state)
                actions.append(np.NaN)
                rewards.append(np.NaN)
                break

            state = next_state

        histories.append({'episode': episode,
                          'timestep': timestep,
                          'state_0': [s[0] for s in states],
                          'state_1': [s[1] for s in states],
                          'action': actions,
                          'reward': rewards})

        if model is not None:
            print('Total reward:', total_reward)

    env.close()

    # Convert histories to DataFrame
    df = pd.concat([pd.DataFrame(ep) for ep in histories]).reset_index()
    df = df.drop('index', axis=1)
    return df


if __name__ == '__main__':
    df = cartpole()
    print(df[df['episode'] == 0])