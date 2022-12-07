import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from experience_replay import EvaluationReplay
from utils import load_data, load_pretrained, load_txt



def sample_trajectories(dataset, n, min_size):
    """ Samples N trajectories from the dataset with size min size. If episodes exceed
    min_size we truncate the sequence; trajectories below the minimum size are not included
    :param dataset:              DataFrame from which to extract trajectories
    :param n:                    Number of trajectories to run simulation on
    :param min_trajectory_size:  Minimum trajectory size
    :returns:                    Dataframe of sampled trajectories
    """
    lst = []
    for _, trajectory in dataset.groupby('episode'):
        trajectory = trajectory.copy().reset_index(drop=True)
        if len(trajectory) >= min_size:
            trajectory = trajectory.head(min_size)
            trajectory['timestep'] = np.arange(min_size)
            lst.append(trajectory)
        if len(lst) >= n:
            break
    return pd.concat(lst, axis=0).reset_index(drop=True)


def get_actions(policy, trajectories, batch_size):
    """ Obtains actions for each history in dataset from policy
    :param policy:        A trained policy network
    :param trajectories:  DataFrame with variable-length patient trajectories
    :param batch_size:    Number of histories to process at once
    :returns:             Tensor of shape (n_states, n_actions)
    """
    # load dataset into replay buffer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay = EvaluationReplay(dataset=trajectories.copy(), device=device)

    with torch.no_grad():
        actions = []
        with tqdm(total=len(trajectories), desc='patience...', position=0, leave=True) as pbar:
            for states in replay.iterate(batch_size):
                probs = torch.argmax(policy(states), dim=1).cpu().detach().numpy()
                actions.append(probs)
                pbar.update(states.size(0))

    return np.concatenate(actions, axis=0)


def plot_counterfactual(model, dataset, num_trajectories, min_trajectory_size, what, when, how, smoothness=1, batch_size=32):
    """ Plots the action of the model given counterfactual manipulations of its input
    :param model:                Trained policy model with encoder
    :param dataset:              DataFrame from which to extract patient trajectories
    :param num_trajectories:     Number of trajectories to run simulation on
    :param min_trajectory_size:  Minimum trajectory size
    :param what:                 Which feature to manipulate
    :param when:                 When to manipulate feature
    :param how:                  By how much to manipulate feature
    :param smoothness:           How smooth to manipulate feature
    :param batch_size:           Number of histories to process concurrently (default: 32)
    """
    # sample patient trajectories
    trajectories = sample_trajectories(dataset=dataset, n=num_trajectories, min_size=min_trajectory_size)
    print('Number of trajectories:', trajectories.episode.nunique())

    # manipulate feature specified by 'where' from time point 'when' with 'how'
    for feat, val in zip(what, how):
        for i in range(min_trajectory_size):
            # smoothly interpolate manipulation
            interp = 1 / (1 + np.exp(-(i - when) / smoothness))
            trajectories.loc[trajectories.timestep == i, feat] += interp * val

    # determine actions of model
    actions = get_actions(policy=model, trajectories=trajectories, batch_size=batch_size)
    iv_action = actions // 5
    vp_action = actions % 5

    # plot trajectories
    plt.figure(figsize=(12, 7))

    ### IV fluids
    ax = plt.subplot(1, 2, 1)
    for _, trajectory in trajectories.groupby('episode', observed=True, sort=False):
        ax.plot(trajectory.timestep, trajectory[what[0]], 'r-', alpha=0.6)
        break
    ax.set_xlabel('Time step')
    ax.set_ylabel('Feature value')

    # plot average response of model
    avg_iv_action = iv_action.reshape(-1, min_trajectory_size).mean(axis=0)
    ax2 = ax.twinx()
    ax2.plot(avg_iv_action, 'b-')
    ax2.set_ylabel('IV fluid dose')
    ax2.set_ylim(0, 4)

    ### Vasopressors
    ax3 = plt.subplot(1, 2, 2)
    for _, trajectory in trajectories.groupby('episode', observed=True, sort=False):
        ax3.plot(trajectory.timestep, trajectory[what[0]], 'r-', alpha=0.6)
        break
    ax3.set_xlabel('Time step')
    ax3.set_ylabel('Feature value')

    # plot average response of model
    avg_vp_action = vp_action.reshape(-1, min_trajectory_size).mean(axis=0)
    ax4 = ax3.twinx()
    ax4.plot(avg_vp_action, 'b-')
    ax4.set_ylabel('Vasopressor dose')
    ax4.set_ylim(0, 4)

    plt.show()



if __name__ == '__main__':
    model = load_pretrained("../results/transformer_nsp_experiment_00001/model.pt")
    dataset = load_data("../../preprocessing/datasets/amsterdam-umc-db_v2/aggregated_full_cohort_2h/test.csv")

    features = load_txt("../../preprocessing/datasets/amsterdam-umc-db_v2/aggregated_full_cohort_2h/state_space_features.txt")

    plot_counterfactual(
        model=model,
        dataset=dataset,
        num_trajectories=2000,
        min_trajectory_size=20,
        # how and what to manipulate?
        what=['x%d' % features.index(x) for x in ['mean_bp', 'sys_bp', 'dias_bp']],
        when=10,
        how=(-2, -2, -2),
        smoothness=0.2,
    )