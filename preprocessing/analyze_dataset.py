import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def print_stats(dfs, labels):
    print('\nDistr. of reward:')
    table = np.zeros((7, len(labels)), dtype=np.int64)
    for i, df in enumerate(dfs):
        table[0][i] = np.sum(len(df.reward))
        table[1][i] = np.sum(df.reward == 0)
        table[2][i] = np.sum(df.reward > 0)
        table[3][i] = np.sum(df.reward < 0)
        table[4][i] = np.min(df.reward)
        table[5][i] = np.mean(df.reward)
        table[6][i] = np.max(df.reward)
    row_labels = ['|R|', '|R = 0|', '|R > 0|', '|R < 0|', 'min(R)', 'mean(R)', 'max(R)']
    print(pd.DataFrame(table, columns=labels, index=row_labels))


def action_support(dfs, labels):
    print('\nSupport of actions (all states):')
    dfs2 = [df.groupby('action').action.size() for df in dfs]
    print(pd.DataFrame({label: lst for label, lst in zip(labels, dfs2)}))

    print('\nSupport of actions (terminal states):')
    dfs2 = [df[df.reward != 0].groupby('action').action.size() for df in dfs]
    print(pd.DataFrame({label: lst for label, lst in zip(labels, dfs2)}))


def expected_reward_actions(dfs, labels):
    print('\nExp. reward actions (all states):')
    dfs2 = [df.groupby('action').reward.mean() for df in dfs]
    print(pd.DataFrame({label: lst for label, lst in zip(labels, dfs2)}))

    print('\nExp. reward actions (terminal states):')
    dfs2 = [df[df.reward != 0].groupby('action').reward.mean() for df in dfs]
    print(pd.DataFrame({label: lst for label, lst in zip(labels, dfs2)}))


def impossible_actions(dfs, labels, actions=(1, 2, 3, 4)):
    table = np.zeros((len(actions), len(labels)), dtype=np.int64)
    for i, df in enumerate(dfs):
        for j, a in enumerate(actions):
            table[j, i] = np.sum(df.action == a)

    print('\n Occurrence of impossible actions:')
    print(pd.DataFrame(table, columns=labels, index=actions))


if __name__ == '__main__':
    datasets = ['roggeveen_4h', 'roggeveen_4h_with_cv']
    path = '../preprocessing/datasets/mimic-iii/%s/mimic-iii_valid.csv'

    # Load datasets as DataFrames 
    dfs = [pd.read_csv(path % folder) for folder in datasets]

    # Drop NaN-reward rows (terminal states and/or intermediate information)
    dfs = [df[df.reward.notna()] for df in dfs]

    print_stats(dfs, labels=datasets)
    impossible_actions(dfs, labels=datasets, actions=(1, 2, 3, 4))
    action_support(dfs, labels=datasets)
    expected_reward_actions(dfs, labels=datasets)

    
    
