import os
import json
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.ndimage import uniform_filter1d

sns.set_theme(style="darkgrid")

COLORS = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']


def load_performance_metrics(path, metric):
    return np.loadtxt(os.path.join(path, metric + '.npy'))


def get_num_episodes(path):
    # Use 'loss.npy' as a proxy to determine number of training episodes completed
    return len(load_performance_metrics(path, 'loss'))


def main(in_dir, out_dir, paths, metric, smooth_over_episodes=500):
    # Plot validation scores over training episodes
    for i, (name, path) in enumerate(paths.items()):
        # Load scores
        full_path = os.path.join(in_dir, path)
        scores = load_performance_metrics(full_path, metric=metric)

        # Create vector of timesteps
        num_episodes = get_num_episodes(full_path)
        episode = np.linspace(0, num_episodes, scores.shape[0])

        plt.plot(episode, scores, color=COLORS[i], label=name, linewidth=1, alpha=0.5)

        # Optionally, smooth scores using box-kernel of x-episodes
        if smooth_over_episodes:
            kernel_size = int(smooth_over_episodes * scores.shape[0] / num_episodes)
            smoothed_scores = uniform_filter1d(scores, kernel_size, mode='nearest')
            plt.plot(episode, smoothed_scores, color=COLORS[i], linewidth=1.7)

    plt.legend()
    plt.ylabel(metric)
    plt.xlabel('Episode')

    # Save to file before showing
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, 'mimic-iii_valid_%s.pdf' % metric))
    plt.show()


if __name__ == '__main__':
    paths = {'Transformer': 'transformer_experiment_00000'}

    metrics = ['loss', 'wis', 'avg_Q_value', 'physician_entropy']
    in_dir = '../results/'
    out_dir = '../results/figures/'

    for metric in metrics:
        main(in_dir, out_dir, paths, metric, smooth_over_episodes=1000)

