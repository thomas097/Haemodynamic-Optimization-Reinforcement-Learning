import os
import torch
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from tqdm import tqdm


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


class PermutationAttribution:
    def __init__(self, n_features, n_timesteps, n_outputs, n_iters=32):
        self._n_features = n_features
        self._n_timesteps = n_timesteps
        self._n_outputs = n_outputs
        self._n_iters = n_iters

    def __call__(self, episode, model):
        episode = episode.repeat(self._n_iters, 1, 1)

        map_ = torch.zeros(size=(self._n_iters, self._n_outputs) + episode.shape[1:])

        with torch.no_grad():
            for t in tqdm(range(self._n_timesteps)):
                for f in range(self._n_features):
                    # Replace single feature with normal Gaussian noise
                    episode2 = episode.clone().float()
                    episode2[:, t, f] = torch.randn(self._n_iters)

                    # Measure response of output neurons
                    map_[:, :, t, f] = model(episode2)

        attr_map = torch.std(map_, dim=0).detach().numpy()

        for o in range(self._n_outputs):
            plt.imshow(attr_map[o])
            plt.show()


if __name__ == '__main__':
    model = load_pretrained("../results/transformer_nsp_pretraining_00008/encoder.pt")
    dataset = pd.read_csv('../../preprocessing/datasets/mimic-iii/aggregated_1h/mimic-iii_valid.csv')
    #features = read_txt('../../preprocessing/datasets/mimic-iii/aggregated_1h/state_space_features.txt')
    timestep = 71

    # Sample episode from dataset
    episode_id = random.choice(dataset.episode.unique())
    episode = dataset[dataset.episode == episode_id].filter(regex='x\d+').values  # only x* columns!
    episode = torch.tensor(episode[:timestep])

    # Compute attribution map!
    pa = PermutationAttribution(n_features=48, n_timesteps=71, n_outputs=24)
    pa(episode, model)
