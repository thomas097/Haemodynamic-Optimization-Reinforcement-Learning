import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="darkgrid")


def plot_action_space(out_path, data_paths, labels, num_actions=25):
    # Assign each action a unique color
    colors = plt.get_cmap(name='hsv', lut=num_actions)

    # Plot action spaces of datasets side by side
    plt.figure(figsize=(12, 4))
    for i, path in enumerate(data_paths):
        df = pd.read_csv(path)
        max_vaso = df['max_vaso']
        iv_fluid = df['total_iv_fluid']
        action = df['action'].values.astype(int)

        plt.subplot(1, 3, i + 1)
        plt.scatter(iv_fluid, max_vaso, c=colors(action))
        plt.xscale('log')
        plt.yscale('log')

        plt.title(labels[i])
        plt.xlabel('Max VP dose (mcg/kg/min)')
        plt.ylabel('Total IV fluid (mL)' if i == 0 else '')

    plt.tight_layout()
    plt.savefig(os.path.join(out_path, 'mimic-iii_action_space.pdf'))
    plt.show()


def print_action_support(data_paths, labels):
    for i, path in enumerate(data_paths):
        df = pd.read_csv(path)
        counts = df.groupby('action', sort=True).action.size().to_frame()

        # Create 5x5 table of IV fluid x vasopressor dosage
        table = np.zeros(25, dtype=np.uint64)
        for action, count in counts.iterrows():
            table[int(action)] = count
        table = table.reshape(5, 5)

        print('\n' + labels[i])
        print(pd.DataFrame(table[::-1], columns=np.arange(5), index=np.arange(5)[::-1]))


if __name__ == '__main__':
    data_paths = ['../../preprocessing/datasets/mimic-iii/roggeveen_4h_with_cv/mimic-iii_train.csv',
                  '../../preprocessing/datasets/mimic-iii/roggeveen_4h_with_cv/mimic-iii_valid.csv',
                  '../../preprocessing/datasets/mimic-iii/roggeveen_4h_with_cv/mimic-iii_test.csv']
    labels = ['Train', 'Valid', 'Test']

    out_path = '../figures/'

    plot_action_space(out_path, data_paths, labels)
    print_action_support(data_paths, labels)