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


def get_actions(policy, episodes, batch_size=32):
    """ Obtains actions for each history in dataset from policy
    :param policy:      A trained policy network
    :param episodes:    DataFrame with variable-length patient trajectories
    :param batch_size:  Number of histories to process at once
    :returns:           Tensor of shape (n_states, n_actions)
    """
    # load dataset into replay buffer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay = EvaluationReplay(dataset=episodes.copy(), device=device)

    with torch.no_grad():
        actions = []
        with tqdm(total=len(episodes), desc='Evaluating model', position=0, leave=True) as pbar:
            for states in replay.iterate(batch_size):
                probs = torch.argmax(policy(states), dim=1).cpu().detach().numpy()
                actions.append(probs)
                pbar.update(states.size(0))

    return np.concatenate(actions, axis=0)


def manipulate_episodes(policy, episodes, feature_names, features, labels, amounts, n_times, batch_size=64):
    # run policy over trajectories before manipulation
    horizon = len(episodes) // episodes.episode.nunique()
    pre_actions = get_actions(policy, episodes, batch_size=batch_size).reshape(-1, horizon)
    pre_features = [episodes['x%d' % feature_names.index(f)].values.reshape(-1, horizon) for f in features]

    # manipulate trajectories
    period = horizon // (n_times + 1)
    times = np.arange(1, horizon + 1) % period == 0
    kernel = np.array([1.2, 1.6, 0.8, 0.4, 0.2, 0.1])
    kernel = kernel / np.sum(kernel)
    manip_times = np.convolve(times, kernel, mode='same')

    episode_ids = episodes.episode
    for f, a in zip(features, amounts):
        i = 'x%d' % feature_names.index(f)
        episodes[i] = episodes[i].groupby(episode_ids, observed=True).transform(lambda x: x + (manip_times * a))

    # run policy over trajectories after manipulation
    post_actions = get_actions(policy, episodes, batch_size=batch_size).reshape(-1, horizon)
    post_features = [episodes['x%d' % feature_names.index(f)].values.reshape(-1, horizon) for f in features]

    # plot average action over average manipulated feature
    plot_actions(pre_actions, post_actions, pre_features, post_features, feature_names=labels)


def plot_actions(pre_actions, post_actions, pre_features, post_features, feature_names):
    plt.figure(figsize=(13, 8))

    # Vasopressors
    plt.subplot(2, 1, 1)
    avg_pre_max_vaso = np.mean(pre_actions % 5, axis=0)
    avg_post_max_vaso = np.mean(post_actions % 5, axis=0)
    plt.plot(avg_pre_max_vaso, c='C0', alpha=0.5)
    plt.plot(avg_post_max_vaso, c='C0', label='Max vasopressor')

    # IV Fluids
    avg_pre_iv_fluid = np.mean(pre_actions // 5, axis=0)
    avg_post_iv_fluid = np.mean(post_actions // 5, axis=0)
    plt.plot(avg_pre_iv_fluid, c='C1', alpha=0.5)
    plt.plot(avg_post_iv_fluid, c='C1', label='Total IV fluid')
    plt.legend()

    # (Non-)manipulated trajectories
    plt.subplot(2, 1, 2)

    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for i, f in enumerate(feature_names):
        avg_pre_feature = np.mean(pre_features[i], axis=0)
        avg_post_feature = np.mean(post_features[i], axis=0)
        plt.plot(avg_pre_feature, c=colors[i % 10], alpha=0.5)
        plt.plot(avg_post_feature, c=colors[i % 10], label=f)

    plt.legend()
    plt.show()



if __name__ == '__main__':
    policy = load_pretrained("../results/pretrained_autoencoder_experiment_00000/model.pt")
    dataset = load_data("../../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/test.csv")
    features = load_txt("../../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/state_space_features.txt")

    episodes = select_episodes(dataset, length=32)
    print('Extracted %d episodes' % episodes.episode.nunique())

    episodes = manipulate_episodes(
        policy=policy,
        episodes=episodes,
        feature_names=features,
        features=('dias_bp', 'mean_bp', 'sys_bp'),
        labels=('Diastolic BP', 'Mean BP', 'Systolic BP'),
        amounts=(-1, -1, -1),
        n_times=2,
    )


