import os

# to allow running from terminal without pyCharm :(
import sys
sys.path.append('../../ope')
sys.path.append('../../models/q_learning')
sys.path.append('../../models/attention/nn')
sys.path.append('../../models/ckconv/nn')
sys.path.append('../../models')

import torch
import argparse
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import colors
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


class Gradients:
    def __init__(self, output):
        """ Computes gradients w.r.t. input
        :param n_outputs:     Number of outputs of model
        """
        self._output = output

    def _gradients(self, x, model):
        """ Compute gradient of interpolated inputs w.r.t. each output channel
        :param interp_inputs:  Tensor of interpolated images
        :param model:          Pytorch model
        """

        # prepare differentiable input
        xi = x.unsqueeze(0).detach().clone()
        xi.requires_grad = True

        # compute gradient of input x w.r.t. to output `i`
        model.zero_grad()
        loss = model(xi)[0, self._output]#torch.neg()
        loss.backward()

        return xi.grad.cpu().detach().clone()

    def __call__(self, inputs, model):
        """ Compute attribution map for input x and model w.r.t to each
        output feature of model
        :param inputs:         Input to model of shape (n_timesteps, n_features)
        :param model:          Pytorch model for which to compute attribution maps
        :param plot:           Whether to plot the attribution map
        :returns:              Ndarray of attribution maps, one for each output channel
        """
        return self._gradients(inputs, model).detach().numpy()[0]


def plot(feature_names, select_only, attr_map_iv, attr_map_vp, correct_gamma=4):
    """ Plots attribution maps for each output channel
    :param feature_names:  Names of input features in attribution maps
    :param attr_maps:      Attribution maps of shape (n_timesteps, n_features)
    """
    # color map where the highest magnitude sets min/max of color range
    if correct_gamma:
        attr_map_iv = torch.tanh(correct_gamma * torch.Tensor(attr_map_iv)).detach().numpy()
        attr_map_vp = torch.tanh(correct_gamma * torch.Tensor(attr_map_vp)).detach().numpy()

    # limit plot to only those features selected
    idx = [i for i, name in enumerate(feature_names) if name in select_only]
    attr_map_iv = -attr_map_iv[:, idx]
    attr_map_vp = -attr_map_vp[:, idx]

    # Color ranges
    rng_iv = np.max(np.abs(attr_map_iv))
    rng_vp = np.max(np.abs(attr_map_vp))
    divnorm_iv = colors.TwoSlopeNorm(vmin=-rng_iv, vcenter=0., vmax=rng_iv)
    divnorm_vp = colors.TwoSlopeNorm(vmin=-rng_vp, vcenter=0., vmax=rng_vp)

    # PLOT IV
    fig = plt.figure(figsize=(6, 3.2))
    ax = fig.add_subplot(121)
    ax.imshow(attr_map_iv.T, cmap='bwr', norm=divnorm_iv)

    # labels
    n_timesteps, n_features = attr_map_iv.shape
    ax.set_xlabel('Time step')
    ax.set_title('IV Fluid')
    ax.set_ylabel('Input feature')
    ax.yaxis.tick_right()
    ax.set_yticks([])
    ax.yaxis.set_ticks_position('right')

    # grid lines
    ax.set_xticks(np.arange(-.5, n_timesteps, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_features, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    ## PLOT VP
    ax = fig.add_subplot(122)
    ax.imshow(attr_map_vp.T, cmap='bwr', norm=divnorm_vp)

    # labels
    ax.set_xlabel('Time step')
    ax.set_title('Vasopressor')
    ax.yaxis.tick_right()
    ax.set_yticks(np.arange(n_features))
    ax.set_yticklabels(select_only)
    ax.yaxis.set_ticks_position('right')

    # grid lines
    ax.set_xticks(np.arange(-.5, n_timesteps, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_features, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    plt.tight_layout()
    plt.show()


class ToLinearDose(torch.nn.Module):
    """ Maps Q-values to continuous doses of fluids and vasopressors (in a
        differentiable way to ensure we can backprop through this) """
    def __init__(self, ivs, vps):
        super().__init__()
        # Weights for each combination of vasopressors and doses
        ivs = torch.repeat_interleave(torch.FloatTensor(ivs), repeats=5, dim=0).unsqueeze(0)
        vps = torch.FloatTensor(vps).repeat(5).unsqueeze(0)
        self._doses = torch.concat([ivs, vps], dim=0).double()

    def forward(self, q_values, tau=2):
        strengths = torch.softmax(tau * q_values, dim=1)[0]
        return torch.matmul(self._doses, strengths).unsqueeze(0)


class FirstLayer(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x[:, -1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This code creates an attribution map using one input gradients')
    parser.add_argument('-d', "--dataset", type=str, default='amsterdam-umc-db')
    parser.add_argument('-m', "--model", type=str, default='last_state')
    parser.add_argument('-p', "--partition", type=str, default='valid')
    parser.add_argument('-n', "--num_histories", type=int, default=500)
    parser.add_argument('-t', "--truncate", type=int, default=12)
    args = vars(parser.parse_args())
    print('Running with args:', args)

    model = load_pretrained('../results/attribution/%s-%s.pt' % (args['dataset'], args['model']))
    data = pd.read_csv('../../preprocessing/datasets/%s/aggregated_full_cohort_2h/%s.csv' % (args['dataset'], args['partition']))
    features = read_txt('../../preprocessing/datasets/%s/aggregated_full_cohort_2h/state_space_features.txt' % args['dataset'])
    N = 500

    # Which ones to plot (found empirically to be most important across policies/datasets)
    select_only = ['max_vaso_prev', 'total_iv_fluid_prev', 'ventilator', 'sirs_score', 'weight', 'mean_bp', 'dias_bp',
                   'sys_bp', 'aptt', 'lactate', 'creatinine', 'hb', 'hematocrit', 'art_ph']
    select_only_labels = ['Prev. Vasopressors', 'Prev. Fluids', 'Ventilator (on/off)', 'SIRS score', 'Weight', 'Mean BP',
                          'Dias. BP', 'Sys. BP', 'aPTT', 'Lactate', 'Creatinine', 'Hemoglobin', 'Hematocrit', 'Arterial pH']

    # Reproducibility
    random.seed(123)

    # Augment model with mapping from discrete actions to vasopressor and IV doses
    to_linear_dose = ToLinearDose(ivs=np.array([0, 13.005, 75.885, 151.9, 467.238]) / 500, # median dose of each bin
                                  vps=np.array([0, 0.045, 0.160, 0.342, 0.670]))
    # to_linear_dose = ToLinearDose(ivs=np.array([0, .25, .5, .75, 1.0]),  # median dose of each bin
    #                               vps=np.array([0, .25, .5, .75, 1.0]))
    model = torch.nn.Sequential(model, to_linear_dose)

    # To compute gradients w.r.t. each treatment separately
    ig_iv = Gradients(output=0)
    ig_vp = Gradients(output=1)

    attr_maps_iv = []
    attr_maps_vp = []

    for i in tqdm(range(args['num_histories'])):

        # Gather episodes of length 8 (truncate if longer)
        episode = []
        while len(episode) < args['truncate']:
            # Sample episode from dataset
            episode_id = random.choice(data.episode.unique())
            episode = data[data.episode == episode_id].filter(regex='x\d+').values  # only x* columns!
            episode = torch.tensor(episode[-args['truncate']:]).float()

        # Compute attribution map!
        attr_maps_iv.append(ig_iv(episode, model))
        attr_maps_vp.append(ig_vp(episode, model))

    # Compute average map over histories
    attr_map_iv = np.mean(attr_maps_iv, axis=0)
    attr_map_vp = np.mean(attr_maps_vp, axis=0)

    plot(feature_names=features, select_only=select_only, attr_map_iv=attr_map_iv, attr_map_vp=attr_map_vp)
