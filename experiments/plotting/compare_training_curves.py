import os
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


def load_performance_metrics(path, metric):
    return np.loadtxt(os.path.join(path, metric + '.npy'))


def num_episodes(path):
    # Use 'loss.npy' to determine number of training episodes
    return len(load_performance_metrics(path, 'loss'))


def main(paths, metric):
    for name, path in paths.items():
        score = load_performance_metrics(path, metric=metric)
        episode = np.linspace(0, num_episodes(path), score.shape[0])
        plt.plot(episode, score, label=name, linewidth=1.7, alpha=0.7)
    plt.legend()
    plt.ylabel(metric)
    plt.xlabel('Episode')
    plt.show()


if __name__ == '__main__':
    metric = 'wis'
    paths = {'CKCNN': '../results/ckcnn_experiment_2022-10-23_20-55-07',
             'Roggeveen et al.': '../results/roggeveen_experiment_2022-10-23_20-44-38'}
    main(paths, metric)

