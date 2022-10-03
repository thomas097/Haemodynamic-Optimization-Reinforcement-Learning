import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


def load_data(path, icustay_id, feature):
    df = pd.read_csv(path, usecols=['icustay_id', 'timestep', feature])
    df = df[df['icustay_id'] == icustay_id]
    df['timestep'] = pd.to_datetime(df['timestep'])
    return df


if __name__ == '__main__':
    DATA_DIR = 'datasets/mimic-iii/'
    SET = 'train'
    FEATURE = 'resp_rate'
    ICUSTAY_ID = 279529

    # Load icustay_ids of respective data set
    icustay_ids = np.loadtxt(DATA_DIR + 'handcrafted/mimic-iii_icustay_ids_%s.npy' % SET)

    # Load handcrafted (Roggeveen et al.) and simple dataset restricted to FEATURE
    df1 = load_data(DATA_DIR + 'handcrafted/mimic-iii_%s_handcrafted.csv' % SET, ICUSTAY_ID, FEATURE)
    df2 = load_data(DATA_DIR + 'simple/mimic-iii_%s_simple.csv' % SET, ICUSTAY_ID, FEATURE)

    # Z-score signals to make sure their normalization is identical
    y1 = (df1[FEATURE] - np.mean(df1[FEATURE])) / np.std(df1[FEATURE])
    y2 = (df2[FEATURE] - np.mean(df2[FEATURE])) / np.std(df2[FEATURE])

    # Plot
    plt.plot(df1['timestep'], y1, '.-', alpha=0.5, label='Roggeveen et al. (4h)')
    plt.plot(df2['timestep'], y2, '^-', alpha=0.5, label='Ours (1h)')
    plt.legend()
    plt.show()


