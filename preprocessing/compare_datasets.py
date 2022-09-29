import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


if __name__ == '__main__':
    PART = 'train'
    FEATURE1 = 'max_vp'
    FEATURE2 = 'max_vaso'

    # Load icustay_ids of respective dataset
    icustay_ids = np.loadtxt('mimic-iii_icustay_ids_%s.npy' % PART)

    # Load binned dataset (Roggeveen et al.) restricted to FEATURE
    df1 = pd.read_csv('mimic-iii_%s_handcrafted.csv' % PART, usecols=['icustay_id', 'timestep', FEATURE1])
    df1['timestep'] = pd.to_datetime(df1['timestep'])

    # Load non-binned dataset restricted to FEATURE
    df2 = pd.read_csv('mimic-iii_%s_no-binning.csv' % PART)
    df2 = df2[df2['item_id'] == FEATURE2]
    df2['timestep'] = pd.to_datetime(df2['timestep'])

    while True:
        # Select data from random ICU stay
        icustay_id = random.choice(icustay_ids)
        df1_icustay = df1[df1['icustay_id'] == icustay_id]
        df2_icustay = df2[df2['icustay_id'] == icustay_id]

        # Plot
        plt.plot(df1_icustay['timestep'], df1_icustay[FEATURE1], label='Roggeveen et al. (binned)')
        plt.plot(df2_icustay['timestep'], df2_icustay['value'], label='Ours (non-binned)')
        plt.legend()
        plt.show()


