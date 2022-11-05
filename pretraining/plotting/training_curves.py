import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import uniform_filter1d

sns.set_theme(style="darkgrid")


def load_config(path):
    with open(path, 'r') as file:
        return json.load(file)


def plot_training_curve(path, smooth_over_episodes):
    config = load_config(os.path.join(path, 'config.json'))
    train_loss = np.loadtxt(os.path.join(path, 'train_loss.npy'))
    valid_loss = np.loadtxt(os.path.join(path, 'valid_loss.npy'))
    valid_f1 = np.loadtxt(os.path.join(path, 'valid_f1.npy'))

    # Determine number of batches per episode
    num_episodes = len(valid_loss) * config['experiment']['eval_after']
    batches_per_episode = int(round(len(train_loss) / num_episodes))

    # Smooth over episodes
    train_loss = uniform_filter1d(train_loss, size=batches_per_episode * smooth_over_episodes)

    # Plot training curve
    train_x = np.linspace(0, num_episodes, len(train_loss))
    plt.plot(train_x, train_loss, label='CE loss (train)')

    # Plot validation curves
    valid_x = np.linspace(0, num_episodes, len(valid_loss))
    plt.plot(valid_x, valid_loss, label='CE loss (valid)')
    plt.plot(valid_x, valid_f1, label='F1 (valid)')

    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    path = "../results/transformer_v2_pretraining_00000"
    plot_training_curve(path, smooth_over_episodes=3)

