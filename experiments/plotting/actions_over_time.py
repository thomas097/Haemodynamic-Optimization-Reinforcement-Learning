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
sns.set_theme(style="darkgrid")


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


def action_to_iv_vp_bins(actions, bin_file):
    """ Map list of actions 0 - 24 to their respective IV/VP bins 0 - 4 """
    # extract iv/vp bin (0-4) for each action (0-24)
    iv_bins = np.array([bin_file[a][0] for a in actions])
    vp_bins = np.array([bin_file[a][1] for a in actions])
    return iv_bins, vp_bins


def action_proportions_at_timesteps(action_bins, n_actions=5):
    """ For each timestep determine proportion of each action """
    n_timesteps = action_bins.shape[1]
    action_props = []
    for t in range(n_timesteps):
        counts = np.array([np.sum(action_bins[:, t] == a) for a in range(n_actions)])
        action_props.append(counts / np.sum(counts))
    return np.array(action_props)


def actions_over_time(encoder, policy, dataset, bin_file, n_episodes, truncate=256, batch_size=8):
    """ determines proportion of IV/VP actions (0 - 4) chosen by model over time
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_encoder = encoder is not None

    # move dataset (limited to a random selection of n_episodes) to replay buffer
    episodes = random.sample(list(dataset.episode.unique()), k=n_episodes)
    dataset = dataset[dataset.episode.isin(episodes)]
    replay_buffer = EvaluationReplay(dataset, device=device, max_len=truncate, return_history=use_encoder)

    # compute total number of 'actionable' states in dataset
    states = dataset[dataset.action.notna()]
    total_states = states.shape[0]

    # Collect actions by model
    actions = []
    with torch.no_grad():
        with tqdm(total=total_states) as pbar:
            for x in replay_buffer.iterate(batch_size):
                # feed batch of states (x) through model to predict action
                h = encoder(x).detach() if use_encoder else x.detach()
                a = torch.argmax(policy(h), axis=1)
                actions.extend(a.tolist())

                pbar.update(x.size(0))

    # determine which IV/VP action (0 - 4) was takes at each timestep in each episode
    iv_bins, vp_bins = action_to_iv_vp_bins(actions, bin_file=bin_file)

    # create table of shape (n_episodes, n_timesteps)
    n_timesteps = states.episode.value_counts().values[0]
    iv_bins = iv_bins.reshape(n_episodes, n_timesteps)
    vp_bins = vp_bins.reshape(n_episodes, n_timesteps)

    # determine proportions of IV/VP at each time step
    iv_action_props = action_proportions_at_timesteps(iv_bins)
    vp_action_props = action_proportions_at_timesteps(vp_bins)
    return iv_action_props, vp_action_props


def plot(iv_action_props, vp_action_props, iv_labels, vp_labels):
    # separate IV/VP levels
    t = np.arange(vp_action_props.shape[0])
    ivs = [iv_action_props[:, i] for i in range(iv_action_props.shape[1])]
    vps = [vp_action_props[:, i] for i in range(vp_action_props.shape[1])]

    # Reverse order so that 'No-IV' is on top (visually makes most sense)
    ivs = ivs[::-1]
    vps = vps[::-1]
    iv_labels = iv_labels[::-1]
    vp_labels = vp_labels[::-1]

    # IV fluids
    plt.figure(figsize=(12, 5))
    ax0 = plt.subplot(121)
    ax0.stackplot(t, *ivs, labels=iv_labels)
    ax0.set_title('IV fluid administration')
    ax0.set_xlabel('Time step')
    ax0.set_ylabel('% patients')

    # add legend to bottom of frame
    handles, labels = ax0.get_legend_handles_labels()
    ax0.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, prop={'size': 9})

    # vasopressors
    ax1 = plt.subplot(122)
    ax1.stackplot(t, *vps, labels=vp_labels)
    ax1.set_title('Vasopressor administration')
    ax1.set_xlabel('Time step')
    ax1.set_ylabel('% patients')

    # add legend to bottom of frame
    handles, labels = ax1.get_legend_handles_labels()
    ax1.legend(handles[::-1], labels[::-1], loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, prop={'size': 9})

    # to move legend back into view
    plt.subplots_adjust(bottom=0.17)
    plt.show()


def physician_actions_over_time(dataset, bin_file):
    """ Plots actions of physician over time """
    # determine which IV/VP action (0 - 4) was takes at each timestep in each episode
    dataset = dataset[dataset.action.notna()]
    iv_bins, vp_bins = action_to_iv_vp_bins(dataset.action, bin_file=bin_file)

    # create table of shape (n_episodes, n_timesteps)
    n_episodes = len(dataset.episode.unique())
    n_timesteps = dataset.episode.value_counts().values[0]
    iv_bins = iv_bins.reshape(n_episodes, n_timesteps)
    vp_bins = vp_bins.reshape(n_episodes, n_timesteps)

    # determine proportions of IV/VP at each time step
    iv_action_props = action_proportions_at_timesteps(iv_bins)
    vp_action_props = action_proportions_at_timesteps(vp_bins)
    return iv_action_props, vp_action_props


if __name__ == '__main__':
    encoder = load_pretrained('../results/transformer_v2_experiment_00001/encoder.pt')
    policy = load_pretrained('../results/transformer_v2_experiment_00001/policy.pt')
    dataset = load_csv('../../preprocessing/datasets/mimic-iii/non_aggregated_1h/mimic-iii_valid.csv')
    bin_file = load_pickle('../../preprocessing/datasets/mimic-iii/non_aggregated_1h/action_to_vaso_fluid_bins.pkl')

    # physician
    iv_action_props, vp_action_props = physician_actions_over_time(dataset, bin_file=bin_file)
    plot(
        iv_action_props=iv_action_props,
        vp_action_props=vp_action_props,
        iv_labels=['No IV', 'IV1', 'IV2', 'IV3', 'IV4'],
        vp_labels=['No VP', 'VP1', 'VP2', 'VP3', 'VP4'],
    )

    # encoder + policy
    iv_action_props, vp_action_props = actions_over_time(
        encoder=encoder,
        policy=policy,
        dataset=dataset,
        bin_file=bin_file,
        n_episodes=500,
        truncate=256,
        batch_size=12 # 72 / 12 minimizes padding!
    )
    plot(
        iv_action_props=iv_action_props,
        vp_action_props=vp_action_props,
        iv_labels=['No IV', 'IV1', 'IV2', 'IV3', 'IV4'],
        vp_labels=['No VP', 'VP1', 'VP2', 'VP3', 'VP4'],
    )

