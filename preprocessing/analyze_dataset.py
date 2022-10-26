import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


def print_stats(dfs, labels):
    print('\nDistr. of reward:')
    table = np.zeros((6, len(labels)), dtype=np.int64)
    for i, df in enumerate(dfs):
        table[0][i] = np.sum(df.reward == 0)
        table[1][i] = np.sum(df.reward > 0)
        table[2][i] = np.sum(df.reward < 0)
        table[3][i] = np.min(df.reward)
        table[4][i] = np.mean(df.reward)
        table[5][i] = np.max(df.reward)
    row_labels = ['R = 0', 'R > 0', 'R < 0', 'min(R)', 'mean(R)', 'max(R)']
    print(pd.DataFrame(table, columns=labels, index=row_labels))
        

def expected_reward_actions(dfs, labels):
    print('\nExp. reward actions (all states):')
    dfs2 = [df.groupby('action').reward.mean() for df in dfs]
    print(pd.DataFrame({label: lst for label, lst in zip(labels, dfs2)}))

    print('\nExp. reward actions (terminal states):')
    dfs2 = [df[df.reward != 0].groupby('action').reward.mean() for df in dfs]
    print(pd.DataFrame({label: lst for label, lst in zip(labels, dfs2)}))
    


if __name__ == '__main__':
    datasets = ['roggeveen_4h', 'attention']
    path = '../preprocessing/datasets/mimic-iii/%s/mimic-iii_valid.csv'

    # Load datasets as DataFrames 
    dfs = [pd.read_csv(path % folder) for folder in datasets]

    # Drop NaN-reward rows (terminal states and/or intermediate information)
    dfs = [df[df.reward.notna()] for df in dfs]

    print_stats(dfs, labels=datasets)
    expected_reward_actions(dfs, labels=datasets)

    
    
