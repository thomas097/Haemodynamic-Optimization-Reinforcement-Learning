import gym
import numpy as np
import pandas as pd


def discrete_pendulum(model=None, num_episodes=50, max_timesteps=1000):
    """ Runs a model on the Pendulum-v1 gym environment or generates
        off-policy samples to train model on (when model=None). In case
        no model is provided, a uniform random behavior policy is used.
        This function is primarily meant as a benchmark for development.
    """
    env = gym.make("Pendulum-v0")

    histories = []

    for ep in range(num_episodes):
        state = env.reset()
        total_reward = 0
        episode, timestep, states, actions, rewards = [], [], [], [], []

        for ts in range(max_timesteps):
            if model is not None:
                action = model.sample(state)
            else:
                action = round(env.action_space.sample()[0]) + 2  # ~ [0, 4]

            next_state, reward, terminated, info = env.step([action - 2])
            total_reward += reward

            episode.append(ep)
            timestep.append(ts)
            states.append(state)
            actions.append(action)
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
                          'state_2': [s[2] for s in states],
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
    df = discrete_pendulum()
    print(df[df['episode'] == 0])