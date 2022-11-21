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


class IntegratedGradients:
    def __init__(self, n_alphas, n_baselines, n_outputs):
        """ Implementation of Integrated Gradients (Sundararajan et al., 2017)
        For details, see: https://arxiv.org/abs/1703.01365
        :param n_alphas:      Number of interpolations of the input to use
        :param n_baselines:   Number of normal baselines to sample
        :param n_outputs:     Number of outputs of model
        """
        self._n_alphas = n_alphas
        self._n_baselines = n_baselines
        self._n_outputs = n_outputs

    def _interpolate_input(self, inputs, baseline):
        """ Interpolates between inputs and baseline in n equal steps defined by n_alphas
        :param inputs:    Input to model of shape (n_timesteps, n_features)
        :param baseline:  Baseline of shape (n_timesteps, n_features)
        :returns:         Tensor of shape (n_alphas, n_timesteps, n_features) of interpolated inputs
        """
        inputs = inputs.unsqueeze(0)
        baseline = baseline.unsqueeze(0)
        alphas = torch.linspace(0, 1, self._n_alphas).unsqueeze(1).unsqueeze(1)
        delta = inputs - baseline
        return baseline + alphas * delta

    def _gradients(self, interp_inputs, model, n):
        """ Compute gradient of interpolated inputs w.r.t. each output channel
        :param interp_inputs:  Tensor of interpolated images
        :param model:          Pytorch model
        :param n:              Number of baseline
        :returns:              List with tuples containing integer indices of output channels and
                               a matching stack of gradient tensors for each interpolated input
        """
        grads = defaultdict(list)

        # process each interpolation step separately
        # PyTorch does not support by-channel gradients :(
        for x in tqdm(interp_inputs, desc='Baseline %d' % n):
            for i in range(self._n_outputs):
                # prepare differentiable input
                xi = x.unsqueeze(0).detach().clone()
                xi.requires_grad = True

                # compute gradient of input x w.r.t. to output `i`
                model.zero_grad()
                loss = torch.neg(model(xi)[0, i])
                loss.backward()

                grad = xi.grad.cpu().detach().clone()
                grads[i].append(grad)

        # recombine stack of gradients
        return [(i, torch.concat(g, dim=0)) for i, g in grads.items()]

    def _plot(self, feature_names, attr_maps):
        """ Plots attribution maps for each output channel
        :param feature_names:  Names of input features in attribution maps
        :param attr_maps:      Attribution maps of shape (n_timesteps, n_features)
        """
        for i, attr_map in attr_maps.items():
            # color map where the highest magnitude sets min/max of color range
            rng = np.max(np.abs(attr_map))
            divnorm = colors.TwoSlopeNorm(vmin=-rng, vcenter=0., vmax=rng)

            # plot
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111)
            ax.imshow(attr_map.T, cmap='bwr', norm=divnorm)

            # labels
            n_timesteps, n_features = attr_map.shape
            ax.set_xlabel('Time step')
            ax.set_ylabel('Input feature')
            ax.set_title('Output channel %d' % i)
            ax.yaxis.tick_right()
            ax.set_yticks(np.arange(n_features))
            ax.set_yticklabels(feature_names)
            ax.yaxis.set_ticks_position('right')

            # grid lines
            ax.set_xticks(np.arange(-.5, n_timesteps, 1), minor=True)
            ax.set_yticks(np.arange(-.5, n_features, 1), minor=True)
            ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

            plt.tight_layout()
            plt.show()

    def __call__(self, inputs, model, plot=True, feature_names=()):
        """ Compute attribution map for input x and model w.r.t to each
        output feature of model
        :param inputs:         Input to model of shape (n_timesteps, n_features)
        :param model:          Pytorch model for which to compute attribution maps
        :param plot:           Whether to plot the attribution map
        :param feature_names:  List of input feature names (used when plot=True)
        :returns:              Ndarray of attribution maps, one for each output channel
        """
        # collect attribution maps over baselines in defaultdict
        attr_maps = defaultdict(list)

        for n in range(self._n_baselines):
            # sample normally-distributed baseline
            baseline = torch.randn(inputs.shape).to(inputs.device)

            # interpolate input with baseline in a number of steps
            interp_inputs = self._interpolate_input(inputs, baseline).float()

            # compute gradients w.r.t. outputs
            for channel, grads in self._gradients(interp_inputs, model, n=n):

                # Riemann's approximation of area under x_grad
                avg_grad = torch.mean((grads[:-1] + grads[1:]) / 2, dim=0)

                # scale w.r.t. input
                ig_map = (inputs - baseline) * avg_grad
                attr_maps[channel].append(ig_map.unsqueeze(0))

        # average maps over baselines
        attr_maps = {i: torch.mean(torch.concat(maps), dim=0) for i, maps in attr_maps.items()}

        if plot:
            self._plot(feature_names, attr_maps)

        return attr_maps


if __name__ == '__main__':
    model = load_pretrained("../results/transformer_nsp_pretraining_00001/encoder.pt")
    dataset = pd.read_csv('../../preprocessing/datasets/mimic-iii/aggregated_all_1h/mimic-iii_valid.csv')
    features = read_txt('../../preprocessing/datasets/mimic-iii/aggregated_all_1h/state_space_features.txt')
    timestep = 71

    # Sample episode from dataset
    episode_id = random.choice(dataset.episode.unique())
    episode = dataset[dataset.episode == episode_id].filter(regex='x\d+').values  # only x* columns!
    episode = torch.tensor(episode[:timestep])

    # Compute attribution map!
    ig = IntegratedGradients(n_alphas=32, n_baselines=32, n_outputs=16)
    ig(episode, model, plot=True, feature_names=features)
