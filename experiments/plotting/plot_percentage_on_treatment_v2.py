import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from utils import *

sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize': (12, 4)})


def percentage_on_treatment(actions, num_timesteps, bins_to_action):
    actions = np.array([bins_to_action[a] for a in actions])  # Convert from 5x5 actions to IV or VP
    actions = actions.reshape(-1, num_timesteps)[:, :-1]      # -> admissions x timesteps
    return np.mean(actions > 0, axis=0)


def main(in_dir, out_dir, model_paths, dataset_file, action_bin_file):
    dataset = pd.read_csv(dataset_file)

    # Infer timesteps T from dataset
    num_timesteps = dataset.groupby('episode').size().max()
    assert dataset.groupby('episode').size().mad() == 0  # Sanity check: each episode has same horizon

    # Mapping from 0-25 to 5x5 action space
    action_to_bins = load_actions_to_bins(action_bin_file)
    action_to_iv = {a: x[0] for a, x in action_to_bins.items()}
    action_to_vp = {a: x[1] for a, x in action_to_bins.items()}

    # Estimate proportion on treatment for physician policy
    iv_over_time = [percentage_on_treatment(dataset['action'].values, num_timesteps, action_to_iv)]
    vp_over_time = [percentage_on_treatment(dataset['action'].values, num_timesteps, action_to_vp)]
    labels = ['Physician policy']

    # Estimate proportion on treatment for model policies
    for model_name, model_path in model_paths.items():
        policy = load_pretrained(os.path.join(in_dir, model_path), 'policy.pkl')
        encoder = load_pretrained(os.path.join(in_dir, model_path), 'encoder.pkl')

        # Create matrix of actions prescribed by policy
        model_actions = evaluate_on_dataset(encoder, policy, dataset, _type='actions')
        iv_model = percentage_on_treatment(model_actions, num_timesteps, action_to_iv)
        vp_model = percentage_on_treatment(model_actions, num_timesteps, action_to_vp)

        iv_over_time.append(iv_model)
        vp_over_time.append(vp_model)
        labels.append(model_name)

    # Plot IV and VP in separate plots
    fig, (ax0, ax1) = plt.subplots(1, 2)
    for label, iv, vp in zip(labels, iv_over_time, vp_over_time):
        ax0.plot(iv, label=label)
        ax1.plot(vp, label=label)

    ax0.set_title('Total IV fluid intake')
    ax1.set_title('Max VP dose')
    ax0.set_ylabel('Rate of treatment')
    ax0.set_xlabel('Timestep (t)')
    ax1.set_xlabel('Timestep (t)')

    plt.tight_layout()
    ax1.legend()

    # Save to file before showing
    out_file = '%s_percentage_on_treatment.pdf' % Path(dataset_file).stem
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    plt.savefig(os.path.join(out_dir, out_file))

    plt.show()


if __name__ == '__main__':
    paths = {'Roggeveen et al.': 'roggeveen_experiment_2022-10-24_22-11-41',
             'CKCNN': 'ckcnn_experiment_2022-10-24_22-30-20'}

    in_dir = '../results/'
    out_dir = '../figures/'

    dataset_file = '../../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_test.csv'
    action_bin_file = '../../preprocessing/datasets/mimic-iii/roggeveen_4h/action_to_vaso_fluid_bins.pkl'

    main(in_dir, out_dir, paths, dataset_file, action_bin_file)
