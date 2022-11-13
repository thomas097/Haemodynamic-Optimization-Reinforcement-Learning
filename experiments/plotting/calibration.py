import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import *

sns.set_theme(style="darkgrid")


def get_terminal_reward(rewards):
    all_rewards = rewards.fillna(0.0)
    terminal_reward = all_rewards[all_rewards != 0].sum()  # Drop zero/NaN rewards
    return rewards * 0 + terminal_reward                   # Set all non-NaN rewards to terminal reward


def qvals_against_survival_rate(qvals, survival, bins=25):
    # Group Q-values into bins of equal size
    bin_edges = np.quantile(qvals.flatten(), q=np.linspace(0, 1, bins + 1))
    labels = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(bins)]  # Center of bin to label groups
    qvals = pd.cut(qvals.flatten(), bins=bin_edges, labels=labels)

    # For each bin, compute survival rate
    survival_rates = [np.mean(survival[qvals == l]) for l in labels]
    return labels, survival_rates


def main(model_paths, dataset_path, bins=50):
    # Load dataset from file
    dataset = pd.read_csv(dataset_path)

    # Convert +/-15 rewards to 0/1 survival label
    rewards = dataset.groupby('episode')['reward'].transform(get_terminal_reward).values
    survival = (rewards + 15) / 30

    # Extract physician's actions from dataset
    actions = dataset['action'].values[:, np.newaxis].astype(np.uint8)

    fig, axs = plt.subplots(1, len(model_paths), figsize=(12, 6))

    for i, (model_name, model_path) in enumerate(model_paths.items()):
        policy = load_pretrained(model_path, 'policy.pkl')
        encoder = load_pretrained(model_path, 'encoder.pkl')

        # Determine Q-values of actions by physician according to DQN
        qvals = evaluate_on_dataset(encoder, policy, dataset, _type='qvals')
        phys_qvals = np.take_along_axis(qvals, actions, axis=1)

        # Drop actions chosen at absorbing terminal states
        phys_qvals = phys_qvals[~np.isnan(rewards)]
        phys_survival = survival[~np.isnan(rewards)]

        # TODO: fix discounting!

        # Right: Plot histogram of Q-values
        axs2 = axs[i].twinx()
        axs2.grid(False)
        axs2.hist(phys_qvals.flatten(), bins=bins, color='gray')
        if i == len(model_paths) - 1:
            axs2.set_ylabel('Q value frequency')

        # Left: Plot correlation between Q-value and survival
        axs[i].plot(*qvals_against_survival_rate(phys_qvals, phys_survival), color='C0')
        if i == 0:
            axs[i].set_ylabel('Survival rate')
        axs[i].set_xlabel('Q value')

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dataset_path = '../../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_test.csv'
    paths = {'CKCNN': '../results/ckcnn_experiment_2022-10-23_21-10-05',
             'Markovian': '../results/roggeveen_experiment_2022-10-23_19-28-28'}
    main(paths, dataset_path)