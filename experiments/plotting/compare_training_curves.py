import os
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="darkgrid")


def load_performance_metrics(path, metric):
    return np.loadtxt(os.path.join(path, metric + '.npy'))


def num_episodes(path):
    # Use 'loss.npy' as a proxy to determine number of training episodes completed
    return len(load_performance_metrics(path, 'loss'))


def main(in_dir, out_dir, paths, metric):
    # Plot validation scores over training episodes
    for name, path in paths.items():
        full_path = os.path.join(in_dir, path)
        score = load_performance_metrics(full_path, metric=metric)
        episode = np.linspace(0, num_episodes(full_path), score.shape[0])
        plt.plot(episode, score, label=name, linewidth=1.7, alpha=0.7)

    plt.legend()
    plt.ylabel(metric)
    plt.xlabel('Episode')

    # Save to file before showing
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, 'mimic-iii_valid_%s.pdf' % metric))
    plt.show()


if __name__ == '__main__':
    paths = {'Roggeveen et al.': 'roggeveen_experiment_00002'}

    metrics = ['loss', 'wis', 'avg_Q_value', 'phys_entropy']
    in_dir = '../results/'
    out_dir = '../figures/'

    for metric in metrics:
        main(in_dir, out_dir, paths, metric)

