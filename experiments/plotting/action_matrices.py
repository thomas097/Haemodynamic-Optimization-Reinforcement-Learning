import os
import torch
import random
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from matplotlib.lines import Line2D
from experience_replay import EvaluationReplay

# replaces ugly matplot theme
sns.set_theme(style="white")


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


def predict_actions(encoder, policy, dataset, n_episodes, truncate=256, batch_size=8):
    """ determines proportion of IV/VP actions (0 - 4) chosen by model over time
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # move dataset (limited to a random selection of n_episodes) to replay buffer
    episodes = list(dataset.episode.unique())
    if n_episodes > 0:
        episodes = random.sample(episodes, k=n_episodes)

    dataset = dataset[dataset.episode.isin(episodes)]
    replay_buffer = EvaluationReplay(dataset, device=device, max_len=truncate)

    # compute total number of 'actionable' states in dataset
    states = dataset[dataset.action.notna()]
    total_states = states.shape[0]

    # Collect actions by model
    actions = []
    with torch.no_grad():
        with tqdm(total=total_states) as pbar:
            for x in replay_buffer.iterate(batch_size):
                # feed batch of states (x) through model to predict action
                h = encoder(x).detach()
                a = torch.argmax(policy(h), axis=1)
                actions.extend(a.tolist())

                pbar.update(x.size(0))
    return actions


def matrix_from_actions(actions, bin_file):
    """ Populates action matrix of physician """
    # sanity check: drop NaNs if any
    actions = np.array(actions)[~np.isnan(actions)]

    action_matrix = np.zeros((5, 5), dtype=np.uint64)
    for a in actions:
        action_matrix[bin_file[a]] += 1

    # invert rows so (0, 0) is bottom left
    return action_matrix[::-1]


def plot_action_matrices(matrices, labels):
    # tick labels
    x_labels = ['VP%d' % a if a > 0 else 'No VP' for a in range(5)]
    y_labels = ['IV%d' % a if a > 0 else 'No IV' for a in reversed(range(5))]

    # color map
    cmap = sns.color_palette("light:b", as_cmap=True)

    plt.figure(figsize=(12, 6))
    for i, action_matrix in enumerate(matrices):
        # plot as heatmap
        plt.subplot(1, len(matrices), i + 1)
        sns.heatmap(action_matrix, cbar=False, square=True, linewidths=0.2, annot=True,
                    fmt='g', cmap=cmap, xticklabels=x_labels, yticklabels=y_labels)

        # formatting
        plt.title(labels[i])
        plt.ylabel('IV dose' if i == 0 else '')
        plt.xlabel('VP dose')
        plt.grid(False)
    plt.show()


if __name__ == '__main__':
    encoder = load_pretrained('../results/transformer_experiment_00000/encoder.pt')
    policy = load_pretrained('../results/transformer_experiment_00000/policy.pt')
    dataset = load_csv('../../preprocessing/datasets/mimic-iii/aggregated_all_1h/mimic-iii_valid.csv')
    bin_file = load_pickle('../../preprocessing/datasets/mimic-iii/aggregated_all_1h/action_to_vaso_fluid_bins.pkl')

    # physician
    phys_action_matrix = matrix_from_actions(dataset.action, bin_file=bin_file)

    # model
    model_actions = predict_actions(
        encoder=encoder,
        policy=policy,
        dataset=dataset,
        n_episodes=-1,
    )
    policy_action_matrix = matrix_from_actions(model_actions, bin_file=bin_file)

    # plot!
    plot_action_matrices(
        matrices=[phys_action_matrix, policy_action_matrix],
        labels=['Physician', 'Transformer (CKCNN idem)']
    )