import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from experience_replay import EvaluationReplay
from utils import load_data, load_txt, load_pretrained


def select_episodes(df, length):
    episodes = []
    for _, episode_df in df.groupby('episode'):
        if len(episode_df) >= length:
            episodes.append(episode_df.head(length))
    return pd.concat(episodes, axis=0)


def get_q_values(policy, dataset, batch_size=32):
    """ Obtains actions for each history in dataset from policy
    :param policy:      A trained policy network
    :param dataset:     DataFrame with variable-length patient trajectories
    :param batch_size:  Number of histories to process at once
    :returns:           Tensor of shape (n_states, n_actions)
    """
    # load dataset into replay buffer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay = EvaluationReplay(dataset=dataset.copy(), device=device)

    with torch.no_grad():
        q_values = []
        with tqdm(total=len(dataset), desc='Evaluating model', position=0, leave=True) as pbar:
            for states, actions in replay.iterate(batch_size, return_actions=True):
                policy_q_values = policy(states)
                actions = actions.unsqueeze(1)
                q_value = torch.take_along_dim(policy_q_values, indices=actions, dim=1).cpu().detach().numpy()
                q_values.append(q_value)
                pbar.update(states.size(0))

    return np.concatenate(q_values, axis=0).flatten()


def calibration_plot(policy, dataset, bins=50, batch_size=64):
    # add eventual outcome to each state
    dataset['outcome'] = dataset.groupby('episode').reward.apply(lambda x: x * 0 + (x > 0).any())

    # add Q-values of model for action of physician to each state
    dataset['q_value'] = get_q_values(policy, dataset, batch_size=batch_size)

    # plot Q-values
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hist(dataset.q_value, bins=bins)

    # discretize Q-values into evenly-sized  (within which to compute mortality rate)
    dataset['q_value_bin'], bin_values = pd.cut(dataset.q_value, bins=bins, labels=np.arange(bins), retbins=True)
    centroids = [bin_values[i:i + 2].mean() for i in range(bins)]

    # for each bin, estimate survival rate
    q_values, survival_rates = [], []
    for bin, bin_df in dataset.groupby('q_value_bin', sort=True):
        q_values.append(centroids[bin])
        survival_rates.append(bin_df.outcome.mean())

    ax2 = ax.twinx()
    ax2.plot(q_values, survival_rates, c='grey')
    plt.show()


if __name__ == '__main__':
    policy = load_pretrained("../results/amsterdam-umc-db/ckcnn_experiment_00000/model.pt")
    dataset = load_data("../../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/train.csv")
    features = load_txt("../../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/state_space_features.txt")

    calibration_plot(policy=policy, dataset=dataset, bins=100)


