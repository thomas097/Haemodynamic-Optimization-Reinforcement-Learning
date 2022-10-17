import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


def main(path, metrics, num_episodes):
    plt.figure(figsize=(12, 8))
    for i, metric in enumerate(metrics):
        plt.subplot(len(metrics), 1, i + 1)
        y = np.loadtxt(os.path.join(path, metric + '.npy'))
        x = np.linspace(0, num_episodes, y.shape[0])  # to account for `eval_after` parameter

        plt.plot(x, y)
        plt.title(metric)

    plt.show()


if __name__ == '__main__':
    PATH = 'roggeveen_experiment_2022-10-17_10-41-45'
    METRICS = ['phys_entropy', 'wis']
    NUM_EPISODES = 100000

    main(PATH, METRICS, NUM_EPISODES)

