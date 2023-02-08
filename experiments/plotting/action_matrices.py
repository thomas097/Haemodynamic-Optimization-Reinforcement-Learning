import os

# to allow running from terminal without pyCharm :(
import sys
sys.path.append('../../models/q_learning')
sys.path.append('../../models/attention/nn')
sys.path.append('../../models/ckconv/nn')
sys.path.append('../../models')

import torch
import argparse
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.colors import FuncNorm
from matplotlib.lines import Line2D
from experience_replay import EvaluationReplay

# replaces ugly matplot theme
sns.set_theme(style="white")
plt.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "Times New Roman"



def load_pretrained(path):
    """ Load pretrained pytorch model from file """
    if not os.path.exists(path):
        raise Exception('%s does not exist' % path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    model.eval()
    return model


def load_csv(path):
    """ Load CSV file using pandas """
    if not os.path.exists(path):
        raise Exception('%s does not exist' % path)
    return pd.read_csv(path)


def load_pickle(path):
    """ Load pickled file """
    if not os.path.exists(path):
        raise Exception('%s does not exist' % path)
    with open(path, 'rb') as file:
        return pickle.load(file)


def predict_actions(model, dataset, n_episodes, truncate=256, batch_size=8):
    """ determines proportion of IV/VP actions (0 - 4) chosen by model over time
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # move dataset (limited to a random selection of n_episodes) to replay buffer
    episodes = list(dataset.episode.unique())
    if n_episodes > 0:
        episodes = random.sample(episodes, k=n_episodes)

    dataset = dataset[dataset.episode.isin(episodes)]
    replay_buffer = EvaluationReplay(dataset, device=device, max_len=truncate)

    # Collect actions by model
    actions = []
    with torch.no_grad():
        with tqdm(total=dataset.shape[0]) as pbar:
            for x in replay_buffer.iterate(batch_size):
                # feed batch of states (x) through model to predict action
                a = torch.argmax(model(x).detach(), axis=1)
                actions.extend(a.tolist())
                pbar.update(x.size(0))
    return actions


def matrix_from_actions(actions, bin_file):
    """ Populates action matrix of physician """
    action_matrix = np.zeros((5, 5), dtype=np.uint64)
    for a in actions:
        action_matrix[bin_file[a]] += 1

    # invert rows so (0, 0) is below
    return action_matrix[::-1]


def plot_action_matrices(matrices, labels):
    # tick labels
    x_labels = ['VP%d' % a for a in range(5)]
    y_labels = ['IV%d' % a for a in range(5)][::-1]

    # color map
    cmap = sns.color_palette("light:#2562b3", as_cmap=True)

    plt.figure(figsize=(3 * len(matrices), 3.5))
    for i, action_matrix in enumerate(matrices):
        # plot as heatmap
        plt.subplot(1, len(matrices), i + 1)
        sns.heatmap(action_matrix, cbar=False, square=True, linewidths=0.2, annot=True, fmt='g',
                    cmap=cmap, xticklabels=x_labels, yticklabels=y_labels, annot_kws={"size": 10})

        # formatting
        plt.title(labels[i])
        plt.ylabel('IV dose' if i == 0 else '')
        plt.xlabel('VP dose')
        plt.grid(False)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This code generates action matrices for all models')
    parser.add_argument('-d', "--dataset", type=str, default='amsterdam-umc-db')
    parser.add_argument('-m', '--models', nargs='+', default=['last_state', 'transformer', 'ckcnn'])
    parser.add_argument('-p', "--partition", type=str, default='test')
    args = vars(parser.parse_args())
    print('Running with args:', args)

    dataset = load_csv('../../preprocessing/datasets/%s/aggregated_full_cohort_2h/%s.csv' % (args['dataset'], args['partition']))
    bin_file = load_pickle('../../preprocessing/datasets/%s/aggregated_full_cohort_2h/action_to_vaso_fluid_bins.pkl' % args['dataset'])

    # get action matrices of physician and models
    action_matrices = [matrix_from_actions(dataset.action, bin_file=bin_file)]

    for model in args['models']:
        actions = predict_actions(
            model=load_pretrained('../results/action_matrices/%s_%s.pt' % (args['dataset'], model)),
            dataset=dataset,
            n_episodes=-1,
        )
        action_matrices.append(matrix_from_actions(actions, bin_file=bin_file))

    # plot!
    plot_action_matrices(
        matrices=action_matrices,
        labels=['Physician'] + [s.replace('_', ' ').title() for s in args['models']]
    )