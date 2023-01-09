import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from experience_replay import EvaluationReplay
from utils import load_data, load_txt, load_pretrained

# replaces ugly matplot theme
sns.set_theme(style="darkgrid")

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


def manipulate_episodes(policy, episodes, feature_names, features, labels, amounts, n_times, batch_size=64, model_name='', plot_features=True):
    # Keep features before manipulation?
    horizon = len(episodes) // episodes.episode.nunique()
    pre_features = [episodes['x%d' % feature_names.index(f)].values.reshape(-1, horizon) for f in features]

    # manipulate trajectories
    period = horizon // (n_times + 1)
    times = np.arange(1, horizon + 1) % period == 0
    kernel = np.array([1.2, 1.6, 0.8, 0.4, 0.2, 0.1]) # manually define smoothing kernel
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
    plot_actions(post_actions, pre_features, post_features, feature_names=labels, model_name=model_name, plot_features=plot_features)


def action_proportions_at_timesteps(action_bins, n_actions=5):
    """ For each timestep determine proportion of each action """
    n_timesteps = action_bins.shape[1]
    action_props = []
    for t in range(n_timesteps):
        counts = np.array([np.sum(action_bins[:, t] == a) for a in range(n_actions)])
        action_props.append(counts / np.sum(counts))
    return np.array(action_props)


def plot_actions(post_actions, pre_features, post_features, feature_names, model_name='', plot_features=True):
    plt.figure(figsize=(13, 8 if plot_features else 4))

    vp_labels = ['VP4', 'VP3', 'VP2', 'VP1', 'No VP']
    iv_labels = ['IV4', 'IV3', 'IV2', 'IV1', 'No IV']
    color_map = ["#cf4c4c", "#cfbb4c", "#8dcf4c", "#4ccf83", "#4c9fcf"]

    # determine proportion of actions at each time step
    t = np.arange(post_actions.shape[1])
    iv_post_action_props = 100 * action_proportions_at_timesteps(post_actions // 5)
    vp_post_action_props = 100 * action_proportions_at_timesteps(post_actions % 5)

    # Vasopressors
    ax0 = plt.subplot(221 if plot_features else 121)
    vps = [vp_post_action_props[:, i] for i in range(vp_post_action_props.shape[1])][::-1] # put action 0 up top (visually makes most sense)
    ax0.stackplot(t, *vps, labels=vp_labels, colors=color_map)
    ax0.set_title('%s - Vasopressor dose' % model_name)
    ax0.set_ylabel('% patients')
    ax0.set_ylim([0, 100])
    ax0.set_xlim([0, vp_post_action_props.shape[0] - 1])

    if plot_features:
        handles, labels = ax0.get_legend_handles_labels()
        ax0.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, prop={'size': 11})

    # IV fluids
    ax1 = plt.subplot(222 if plot_features else 122)
    ivs = [iv_post_action_props[:, i] for i in range(iv_post_action_props.shape[1])][::-1]
    ax1.stackplot(t, *ivs, labels=iv_labels, colors=color_map)
    ax1.set_title('%s - IV fluid dose' % model_name)
    ax1.set_ylabel('% patients')
    ax1.set_ylim([0, 100])
    ax1.set_xlim([0, iv_post_action_props.shape[0] - 1])

    if plot_features:
        handles, labels = ax1.get_legend_handles_labels()
        ax1.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, prop={'size': 11})

    # (Non-)manipulated trajectories under Vasopressors and IV
    if plot_features:
        ax3 = plt.subplot(223)
        ax3.set_xlabel('Time step')
        ax3.set_ylabel('Input value')

        colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
        for i, f in enumerate(feature_names):
            avg_pre_feature = np.mean(pre_features[i], axis=0)
            avg_post_feature = np.mean(post_features[i], axis=0)
            ax3.plot(avg_pre_feature, c=colors[i % 10], alpha=0.5, linestyle='--')
            ax3.plot(avg_post_feature, c=colors[i % 10], label=f)

        handles, labels = ax3.get_legend_handles_labels()
        ax3.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.23), ncol=5, prop={'size': 11})

        ax4 = plt.subplot(224)
        ax4.set_xlabel('Time step')
        ax4.set_ylabel('Input value')

        for i, f in enumerate(feature_names):
            avg_pre_feature = np.mean(pre_features[i], axis=0)
            avg_post_feature = np.mean(post_features[i], axis=0)
            ax4.plot(avg_pre_feature, c=colors[i % 10], alpha=0.5, linestyle='--')
            ax4.plot(avg_post_feature, c=colors[i % 10])

    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    policy = load_pretrained("../results/amsterdam-umc-db/transformer_experiment_00001/model.pt")
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
        amounts=(-1.5, -1.5, -1.5),
        n_times=2,
        plot_features=True,
        model_name='Pretrained CKCNN'
    )


