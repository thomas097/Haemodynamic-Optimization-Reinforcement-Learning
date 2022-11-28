import os
import torch
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_pretrained(path):
    """ Load pretrained pytorch model from file
    :param path:  Path to Transformer instance in pt format
    :returns:     A PyTorch model
    """
    if not os.path.exists(path):
        raise Exception('%s does not exist' % path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    model.eval()
    return model


def read_txt(path):
    """ Load text file from path
    :param path:  Path to file
    :returns:     List of '\n' delimited strings
    """
    with open(path, 'r') as file:
        return [line.strip() for line in file.readlines()]


def simulate_trajectory(start_state, model, time_steps=72, truncate=24):
    states = start_state
    with torch.no_grad():
        # autoregressive bootstrapping from the previously generated states
        for t in range(time_steps - start_state.size(1)):
            next_state = model(states[:, -truncate:].float()).unsqueeze(1)
            states = torch.concat([states, next_state], dim=1)
    return states[0]


if __name__ == '__main__':
    model = load_pretrained("../results/lstm_nsp_pretraining_00000/nsp_regressor.pt")
    dataset = pd.read_csv('../../preprocessing/datasets/mimic-iii/aggregated_all_1h/mimic-iii_train.csv')
    feature_names = read_txt('../../preprocessing/datasets/mimic-iii/aggregated_all_1h/state_space_features.txt')

    for _ in range(100):
        # Sample start state from random episode in dataset
        episode_id = random.choice(dataset.episode.unique())
        episode = dataset[dataset.episode == episode_id].filter(regex='x\d+').values  # only x* columns!
        start_state = torch.tensor(episode[:15]).unsqueeze(0)

        # focus on specific feature
        i = feature_names.index('mean_bp')

        # simulate patient trajectory from start state
        simulated = simulate_trajectory(start_state, model, truncate=24, time_steps=72)

        plt.plot(episode[:, i], linestyle='-', label='true trajectory')
        plt.plot(simulated[:, i], linestyle='--', label='simulated')
        plt.legend()
        plt.show()