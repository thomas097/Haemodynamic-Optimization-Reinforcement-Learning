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


def read_txt(path):
    """ Load text file from path
    :param path:  Path to file
    :returns:     List of '\n' delimited strings
    """
    with open(path, 'r') as file:
        return [line.strip() for line in file.readlines()]


class PermutationAttribution:
    def __init__(self, n_outputs, n_samples=64):
        self._n_outputs = n_outputs
        self._n_samples = n_samples

    @staticmethod
    def _plot(attr_map, out_channel, feature_names):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111)
        ax.imshow(attr_map.T)

        # labels
        n_timesteps, n_features = attr_map.shape
        ax.set_xlabel('Time step')
        ax.set_ylabel('Input feature')
        ax.set_title('Output channel %d' % out_channel)
        ax.yaxis.tick_right()
        ax.set_yticks(np.arange(n_features))
        ax.set_yticklabels(feature_names)
        ax.yaxis.set_ticks_position('right')

        plt.tight_layout()
        plt.show()

    def __call__(self, episode, model, plot=True, feature_names=None):
        # simultaneously process each iteration as a batch
        episode = episode.repeat(self._n_samples, 1, 1)

        # create map to store model outputs into
        _, n_timesteps, n_inputs = episode.shape
        res = torch.zeros(size=(self._n_samples, self._n_outputs, n_timesteps, n_inputs))

        # vary features at each timestep one at a time (keep everything
        # constant except that one feature at that timestep)
        with torch.no_grad():
            with tqdm(total=n_timesteps) as pbar:
                for t in range(n_timesteps):
                    pbar.set_description('%d/%d' % (t + 1, n_timesteps))
                    pbar.update(1)
                    for f in range(n_inputs):
                        # replace feature `f` at time `t` with standard Gaussian noise
                        episode_noised = episode.clone().float()
                        episode_noised[:, t, f] = torch.randn(self._n_samples)

                        # measure response of output neurons
                        res[:, :, t, f] = model(episode_noised)

        # extract contribution as stdev of output varying only one feature at one timestep
        # TODO: change
        attr_map = torch.std(res, dim=0).detach().numpy()

        if plot:
            # optionally plot each attribution map one-by-one
            for o in range(self._n_outputs):
                self._plot(attr_map[o], out_channel=o, feature_names=feature_names)


if __name__ == '__main__':
    model = load_pretrained("../results/transformer_nsp_pretraining_00001/encoder.pt")
    dataset = pd.read_csv('../../preprocessing/datasets/mimic-iii/aggregated_1h/mimic-iii_valid.csv')
    features = read_txt('../../preprocessing/datasets/mimic-iii/aggregated_1h/state_space_features.txt')
    timestep = 71

    # Sample episode from dataset
    episode_id = random.choice(dataset.episode.unique())
    episode = dataset[dataset.episode == episode_id].filter(regex='x\d+').values  # only x* columns!
    episode = torch.tensor(episode[:timestep])

    # Compute attribution map!
    pa = PermutationAttribution(n_outputs=16)
    pa(episode, model, feature_names=features)
