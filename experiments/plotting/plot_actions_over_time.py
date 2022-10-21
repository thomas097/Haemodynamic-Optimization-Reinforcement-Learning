import numpy as np
import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")



def plot_intensity_over_time(icustay_ids, actions, action_to_bins, max_vaso_bins, iv_fluid_bins, policy_name=''):
    # Create matrix of actions over time of shape (num_stays, num_time_steps)
    action_mat = np.array([actions[icustay_ids == i] for i in set(icustay_ids)])

    # Compute bin midpoints
    max_vaso_bin_midpoints = np.array([np.mean(max_vaso_bins[i:i + 2]) for i in range(len(max_vaso_bins) - 1)])
    iv_fluid_bin_midpoints = np.array([np.mean(iv_fluid_bins[i:i + 2]) for i in range(len(iv_fluid_bins) - 1)])

    # Convert IDs in action_mat to average IV-fluid and vasopressor doses
    iv_fluid_bins, max_vaso_bins = np.vectorize(action_to_bins.__getitem__)(action_mat)
    max_vaso_mat = max_vaso_bin_midpoints[max_vaso_bins]
    iv_fluid_mat = iv_fluid_bin_midpoints[iv_fluid_bins]

    # Average intake over patient trajectories
    avg_iv_fluid = np.mean(iv_fluid_mat, axis=0)
    avg_max_vaso = np.mean(max_vaso_mat, axis=0)

    # IV fluid plot
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(np.arange(len(avg_iv_fluid)) * 4 + 4, avg_iv_fluid)
    plt.plot([24, 24], [np.min(avg_iv_fluid), np.max(avg_iv_fluid)], c='grey', linestyle='--')
    plt.ylabel('Avg. IV fluid dose (ml)')
    plt.xlabel('Time step (hours)')

    # Max. vaso plot
    plt.subplot(1, 2, 2)
    plt.plot(np.arange(len(avg_iv_fluid)) * 4 + 4, avg_max_vaso, label='treatment dose')
    plt.plot([24, 24], [np.min(avg_max_vaso), np.max(avg_max_vaso)], c='grey', linestyle='--', label='suspected time\nof sepsis onset')
    plt.ylabel('Avg. max. vasopressor dose (mcg/kg/min)')
    plt.xlabel('Time step (hours)')

    plt.suptitle(policy_name)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Load historical physician data (actions + icustay_ids)
    train_df = pd.read_csv('../datasets/mimic-iii/roggeveen/mimic-iii_train.csv')
    physician_actions = train_df['action'].values
    physician_icustay_ids = train_df['icustay_id'].values

    # Load mapping from actions to IV/vaso treatments
    with open('../datasets/mimic-iii/roggeveen/action_to_vaso_fluid_bins.pkl', 'rb') as file:
        action_to_bins = pkl.load(file)

    max_vaso_bins = np.loadtxt('../datasets/mimic-iii/roggeveen/max_vaso_bins.npy')
    iv_fluid_bins = np.loadtxt('../datasets/mimic-iii/roggeveen/total_iv_fluid_bins.npy')

    # Plot intensity of physician actions as a function of time
    plot_intensity_over_time(icustay_ids=physician_icustay_ids, actions=physician_actions, action_to_bins=action_to_bins,
                             max_vaso_bins=max_vaso_bins, iv_fluid_bins=iv_fluid_bins, policy_name='Physician policy')