import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from utils import *


def value_to_color(values, color1=(36, 123, 160), color2=(255, 100, 79)):
    # Scale values to blend factor
    blend = (values / np.max(values))[:, np.newaxis]

    # Blend matrices of color1 and color2
    color1 = np.array([color1]).repeat(len(values), axis=0) / 255
    color2 = np.array([color2]).repeat(len(values), axis=0) / 255
    return blend * color1 + (1 - blend) * color2


def main(model_path, dataset_path, feature):
    dataset = pd.read_csv(dataset_path)
    encoder = load_pretrained(model_path, 'encoder.pkl')

    states = run_encoder_over_dataset(encoder, dataset)

    tsne = TSNE(n_components=2,
                learning_rate='auto',
                init='random',
                perplexity=3,
                random_state=2,
                verbose=1)
    X = tsne.fit_transform(states)

    # Highlight feature in output space using color gradients
    colors = value_to_color(dataset[feature].values)

    plt.scatter(X[:, 0], X[:, 1], color=colors)
    plt.title('t-SNE plot of state space by %s' % FEATURE_NAME)
    plt.show()


if __name__ == '__main__':
    model_path = '../results/ckcnn_experiment_2022-10-23_21-10-05'
    dataset_file = '../../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_test.csv'

    main(model_path, dataset_file, feature='x1')