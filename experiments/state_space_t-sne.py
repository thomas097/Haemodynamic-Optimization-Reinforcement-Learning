import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


if __name__ == '__main__':
    FEATURE_NAME = 'total_iv_fluid'
    COLOR1 = np.array([1, 0, 0])  # red
    COLOR2 = np.array([0, 1, 0])  # blue
    STATE_SPACE_FEATURES = ['max_vp', 'total_iv_fluid', 'sirs_score', 'sofa_score', 'weight', 'ventilator', 'height',
                            'age', 'gender', 'heart_rate', 'temp', 'mean_bp', 'dias_bp', 'sys_bp', 'resp_rate', 'spo2',
                            'natrium', 'chloride', 'kalium', 'trombo', 'leu', 'anion_gap', 'aptt', 'art_ph', 'asat',
                            'alat', 'bicarbonaat', 'art_be', 'ion_ca', 'lactate', 'paco2', 'pao2', 'hb', 'bilirubin',
                            'creatinine', 'inr', 'ureum', 'albumin', 'magnesium', 'calcium', 'glucose',
                            'total_urine_output']
    df = pd.read_csv('../preprocessing/datasets/mimic-iii/handcrafted/mimic-iii_test_handcrafted.csv', usecols=STATE_SPACE_FEATURES)

    # TODO: pass states through encoder!
    X = df.values

    tsne = TSNE(n_components=2,
                learning_rate='auto',
                init='random',
                perplexity=3,
                #random_state=2,
                verbose=1)
    X_new = tsne.fit_transform(X)

    # Assign high feature values color1 and others color2
    feature_vals = df[FEATURE_NAME].values
    blend = (feature_vals / np.max(feature_vals))[:, np.newaxis]
    color1 = COLOR1[np.newaxis].repeat(len(X), axis=0)
    color2 = COLOR2[np.newaxis].repeat(len(X), axis=0)
    colors = blend * color1 + (1 - blend) * color2

    # Plot results using pyplot
    plt.scatter(X_new[:, 0], X_new[:, 1], color=colors)
    plt.title('t-SNE plot of state space by %s' % FEATURE_NAME)
    plt.show()