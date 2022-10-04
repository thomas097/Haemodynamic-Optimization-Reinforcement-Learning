import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle


def plot_action_matrix(df, max_vaso_column, iv_fluid_column, max_vaso_ticks, iv_fluid_ticks, by='sofa_score', N=3):
    """ Plots action matrices of action stored under `action_column` in df,
        grouped by `by` column in N matrices.
    """
    # Cut data into N bins by `by` column
    df['group'] = pd.cut(df[by], bins=N, precision=2)

    max_value = 0
    heatmaps = []
    for i, (group, group_df) in enumerate(df.groupby('group')):
        # Create matrix of frequencies of action combinations
        heatmap = group_df.groupby([max_vaso_column, iv_fluid_column]).size().unstack(level=0).values

        # Update max action frequency seen and assign group a label
        max_value = max(np.nanmax(heatmap), max_value)
        group_label = '%s $\in$ %s' % (by.replace('_', ' ').title(), group)

        heatmaps.append((heatmap, group_label))

    # plot heatmaps side-by-side
    plt.figure(figsize=(12, 4))
    for i, (heatmap, group_label) in enumerate(heatmaps):
        plt.subplot(1, N, i + 1)

        # Plot `sns.heatmap` without color bar but with values
        sns.heatmap(heatmap, cmap="Blues", linewidth=0.3, vmin=0.0, vmax=max_value,
                    cbar=False, annot=True, fmt='g', cbar_kws={"shrink": .8})

        plt.title(group_label)
        plt.xticks(np.arange(len(max_vaso_ticks)), labels=max_vaso_ticks, rotation=90)
        plt.xlabel('Max VP dose ($\mu$g/kg/min)')

        # Only show IV fluid bin edges at left-most table
        if i == 0:
            plt.yticks(np.arange(len(iv_fluid_ticks)), labels=iv_fluid_ticks, rotation=0)
            plt.ylabel('Total IV fluid (ml)')
        else:
            plt.yticks([])
            plt.ylabel('')

    plt.show()


if __name__ == '__main__':
    # TODO: for now just plot the physician policy
    ACTION = 'discretized_action'
    DATA_PATH = '../preprocessing/datasets/mimic-iii/handcrafted/mimic-iii_train_handcrafted.csv'
    
    ACTIONS_TO_BINS_PATH = '../preprocessing/datasets/mimic-iii/handcrafted/action_to_vaso_fluid_bins.pkl'
    MAX_VASO_BINS_PATH = '../preprocessing/datasets/mimic-iii/handcrafted/max_vaso_bins.npy'
    IV_FLUID_BINS_PATH = '../preprocessing/datasets/mimic-iii/handcrafted/total_iv_fluid_bins.npy'

    # read dataset from csv file
    dataset = pd.read_csv(DATA_PATH)

    # load mapping from discretized actions to vasopressor and IV fluid bins
    with open(ACTIONS_TO_BINS_PATH, 'rb') as file:
        action_to_bins = pickle.load(file)

    max_vaso_bins = np.round(np.loadtxt(MAX_VASO_BINS_PATH), 2)
    iv_fluid_bins = np.round(np.loadtxt(IV_FLUID_BINS_PATH), 2)

    # convert action ids (0 - 24) in `ACTION` column to max vaso (0 - 5) and IV fluid (0 - 5)
    dataset['max_vaso_actions'] = dataset[ACTION].apply(lambda a: action_to_bins[a][0])
    dataset['iv_fluid_actions'] = dataset[ACTION].apply(lambda a: action_to_bins[a][1])

    # assign bins their corresponding labels
    plot_action_matrix(df=dataset,
                       max_vaso_column='max_vaso_actions',
                       iv_fluid_column='iv_fluid_actions',
                       max_vaso_ticks=max_vaso_bins,
                       iv_fluid_ticks=iv_fluid_bins)


