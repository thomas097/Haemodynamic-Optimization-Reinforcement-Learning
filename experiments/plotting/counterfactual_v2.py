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


def manipulate_episodes(policy, episodes, feature_names, features, amounts, n_times, batch_size=64):
    # run policy over trajectories before manipulation
    horizon = len(episodes) // episodes.episode.nunique()
    pre_actions = get_actions(policy, episodes, batch_size=batch_size).reshape(-1, horizon)
    pre_feature = episodes['x%d' % feature_names.index(features[0])].values.reshape(-1, horizon)

    # manipulate trajectories
    period = horizon // n_times
    times = np.arange(1, horizon + 1) % period - 2 == 0
    kernel = np.array([1.6, 0.8, 0.4, 0.2, 0.1])
    kernel = kernel / np.sum(kernel)
    manip_times = np.convolve(times, kernel, mode='same')

    episode_ids = episodes.episode
    for f, a in zip(features, amounts):
        i = 'x%d' % feature_names.index(f)
        episodes[i] = episodes[i].groupby(episode_ids, observed=True).transform(lambda x: x + (manip_times * a))

    # run policy over trajectories after manipulation
    post_actions = get_actions(policy, episodes, batch_size=batch_size).reshape(-1, horizon)
    post_feature = episodes['x%d' % feature_names.index(features[0])].values.reshape(-1, horizon)

    # plot average action over average manipulated feature
    plot_actions(pre_actions, post_actions, pre_feature, post_feature, feature_name=features[0])


def plot_actions(pre_actions, post_actions, pre_feature, post_feature, feature_name):
    plt.figure(figsize=(16, 9))

    # Vasopressors
    avg_pre_max_vaso = np.mean(pre_actions % 5, axis=0)
    avg_post_max_vaso = np.mean(post_actions % 5, axis=0)
    plt.plot(avg_pre_max_vaso, c='C0', linestyle='--', alpha=0.5)
    plt.plot(avg_post_max_vaso, c='C0', label='Max vasopressor')

    # IV Fluids
    avg_pre_iv_fluid = np.mean(pre_actions // 5, axis=0)
    avg_post_iv_fluid = np.mean(post_actions // 5, axis=0)
    plt.plot(avg_pre_iv_fluid, c='C1', linestyle='--', alpha=0.5)
    plt.plot(avg_post_iv_fluid, c='C1', label='Total IV fluid')

    # Manipulated feature
    avg_pre_feature = np.mean(pre_feature, axis=0)
    avg_post_feature = np.mean(post_feature, axis=0)
    plt.plot(avg_pre_feature, c='C2', linestyle='--')
    plt.plot(avg_post_feature, c='C2', label=feature_name)

    plt.legend()
    plt.show()



if __name__ == '__main__':
    policy = load_pretrained("../results/pretrained_transformer_experiment_00001/model.pt")
    dataset = load_data("../../preprocessing/datasets/amsterdam-umc-db_v2/aggregated_full_cohort_2h/test.csv")
    features = load_txt("../../preprocessing/datasets/amsterdam-umc-db_v2/aggregated_full_cohort_2h/state_space_features.txt")

    episodes = select_episodes(dataset, length=24)
    print('Extracted episodes')

    episodes = manipulate_episodes(
        policy=policy,
        episodes=episodes,
        feature_names=features,
        features=('dias_bp', 'mean_bp', 'sys_bp'),
        amounts=(-2, -2, -2),
        n_times=3,
    )


