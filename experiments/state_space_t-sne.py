import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def value_to_color(values, color1=(1, 1, 0), color2=(0, 1, 1)):
    # Scale values to blend factor
    blend = (values / np.max(values))[:, np.newaxis]

    # Blend matrices of color1 and color2
    color1 = np.array([color1]).repeat(len(values), axis=0)
    color2 = np.array([color2]).repeat(len(values), axis=0)
    return blend * color1 + (1 - blend) * color2
    


if __name__ == '__main__':
    ENCODER_PATH = None
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

    if ENCODER_PATH is not None:
        # TODO: load encoder
        # TODO: pass states through encoder!
        pass
    else:
        X = df.values

    tsne = TSNE(n_components=2,
                learning_rate='auto',
                init='random',
                perplexity=3,
                #random_state=2,
                verbose=1)
    X_new = tsne.fit_transform(X)

    colors = value_to_color(df[FEATURE_NAME].values)

    # Plot results using pyplot
    plt.scatter(X_new[:, 0], X_new[:, 1], color=colors)
    plt.title('t-SNE plot of state space by %s' % FEATURE_NAME)
    plt.show()