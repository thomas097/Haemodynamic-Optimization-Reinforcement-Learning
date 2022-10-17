import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


def main(path, metrics, num_episodes):
    plt.figure(figsize=(12, 4))
    for i, metric, title in enumerate(metrics):
        plt.subplot(1, len(metrics), i + 1)
        y = np.loadtxt(os.path.join(path, metric + '.npy'))
        x = np.linspace(0, num_episodes, y.shape[0])  # to account for `eval_after` parameter

        plt.plot(x, y)
        plt.xlabel('Episode')
        plt.ylabel(title)
        plt.title(title)

    plt.show()


if __name__ == '__main__':
    PATH = 'roggeveen_experiment_2022-10-17_10-59-09'
    METRICS = [('phys_entropy', 'Entropy w.r.t physician actions'),
               ('wis', 'WIS')]
    NUM_EPISODES = 100000

    main(PATH, METRICS, NUM_EPISODES)

