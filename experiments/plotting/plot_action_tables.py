import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from pathlib import Path
from utils import *

sns.set_theme(style="darkgrid")
sns.set(rc={'figure.figsize': (12, 4)})


def create_action_matrix(actions, action_to_bins, labels=range(5)):
    # Convert actions (0-24) back to max vaso (0-4) and IV fluid (0-4)
    mat = np.zeros((5, 5), dtype=np.uint32)
    for action in actions:
        iv, vp = action_to_bins[action]
        mat[iv, vp] += 1
    return pd.DataFrame(data=mat, columns=labels, index=labels).iloc[::-1]


def main(in_dir, out_dir, paths, dataset_file, action_bin_file, start_at=0):
    # Drop absorbing terminal states from dataset
    dataset = pd.read_csv(dataset_file)
    dataset = dataset[dataset.reward == 0]

    # Mask out timesteps preceding `start_from`
    timesteps = dataset.groupby('episode')['timestep'].transform(lambda x: np.arange(len(x)))
    timestep_mask = (timesteps >= start_at).values

    # Load mapping from 0-25 to 5x5 action space
    action_to_bins = load_actions_to_bins(action_bin_file)

    # Create action matrix for physician policy
    action_mats = [create_action_matrix(dataset['action'][timestep_mask], action_to_bins)]
    labels = ['Physician policy']

    # Create action matrix for learnt policies
    for model_name, model_path in paths.items():
        policy = load_pretrained(os.path.join(in_dir, model_path), 'policy.pkl')
        encoder = load_pretrained(os.path.join(in_dir, model_path), 'encoder.pkl')

        # Create action matrix of actions prescribed by policy
        model_actions = evaluate_policy_on_dataset(encoder, policy, dataset, _type='actions')[timestep_mask]
        action_mat = create_action_matrix(model_actions, action_to_bins)

        action_mats.append(action_mat)
        labels.append(model_name)

    # Determine max matrix value
    max_value = max([np.max(mat.values) for mat in action_mats])

    plt.Figure(figsize=(16, 3))
    for i, (label, action_mat) in enumerate(zip(labels, action_mats)):

        # Plot action matrix side-by-side
        plt.subplot(1, len(action_mats), i + 1)
        sns.heatmap(action_mat, cmap="Blues", linewidth=0.3, vmin=0.0, vmax=max_value,
                    cbar=False, annot=True, fmt='g', cbar_kws={"shrink": .8})

        # Only show IV fluid label at left-most side
        if i == 0:
            plt.ylabel('Total IV Fluids (ml)')
        else:
            plt.yticks(range(5), '')
        plt.xlabel('Max VP dose (mcg/kg/min)')
        plt.title(label)

    plt.tight_layout()

    # Save to file before showing
    out_file = '%s_action_matrices_start=%s.pdf' % (Path(dataset_file).stem, start_at)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    plt.savefig(os.path.join(out_dir, out_file))
    plt.show()


if __name__ == '__main__':
    paths = {'Roggeveen et al.': 'roggeveen_experiment_00000',
             'CKCNN': 'ckcnn_experiment_00000'}

    in_dir = '../results/'
    out_dir = '../figures/'

    dataset_file = '../../preprocessing/datasets/mimic-iii/roggeveen_4h_with_cv/mimic-iii_valid.csv'
    action_bin_file = '../../preprocessing/datasets/mimic-iii/roggeveen_4h_with_cv/action_to_vaso_fluid_bins.pkl'

    main(in_dir, out_dir, paths, dataset_file, action_bin_file, start_at=0)  # t=6 -> estimated onset of sepsis


