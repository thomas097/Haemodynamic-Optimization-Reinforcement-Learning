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


class IntegratedGradients:
    def __init__(self, n_alphas, n_baselines, output):
        """ Implementation of Integrated Gradients (Sundararajan et al., 2017)
        For details, see: https://arxiv.org/abs/1703.01365
        :param n_alphas:      Number of interpolations of the input to use
        :param n_baselines:   Number of normal baselines to sample
        :param n_outputs:     Number of outputs of model
        """
        self._n_alphas = n_alphas
        self._n_baselines = n_baselines
        self._output = output

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
        grads = []

        # process each interpolation step separately
        # PyTorch does not support by-channel gradients :(
        for x in interp_inputs:
            # prepare differentiable input
            xi = x.unsqueeze(0).detach().clone()
            xi.requires_grad = True

            # compute gradient of input x w.r.t. to output `i`
            model.zero_grad()
            loss = model(xi)[0, self._output]#torch.neg()
            loss.backward()

            grad = xi.grad.cpu().detach().clone()
            grads.append(grad)

        # recombine stack of gradients
        return torch.concat(grads, dim=0)

    def __call__(self, inputs, model):
        """ Compute attribution map for input x and model w.r.t to each
        output feature of model
        :param inputs:         Input to model of shape (n_timesteps, n_features)
        :param model:          Pytorch model for which to compute attribution maps
        :param plot:           Whether to plot the attribution map
        :returns:              Ndarray of attribution maps, one for each output channel
        """
        # collect attribution maps over baselines in defaultdict
        attr_maps = []

        for n in range(self._n_baselines):
            # sample normally-distributed baseline
            baseline = torch.randn(inputs.shape).to(inputs.device)

            # interpolate input with baseline in a number of steps
            interp_inputs = self._interpolate_input(inputs, baseline).float()

            # compute gradients w.r.t. outputs
            grads = self._gradients(interp_inputs, model, n=n)

            # Riemann's approximation of area under x_grad
            avg_grad = torch.mean((grads[:-1] + grads[1:]) / 2, dim=0)

            # scale w.r.t. input
            ig_map = (inputs - baseline) * avg_grad
            attr_maps.append(ig_map.unsqueeze(0))

        # average maps over baselines
        return torch.mean(torch.concat(attr_maps), dim=0).detach().numpy()


def plot(feature_names, select_only, attr_map_iv, attr_map_vp, imname, correct_gamma=4.0):
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
    attr_map_iv = attr_map_iv[:, idx]
    attr_map_vp = attr_map_vp[:, idx]

    # Color ranges
    rng_iv = np.max(np.abs(attr_map_iv)) * 0.4
    rng_vp = np.max(np.abs(attr_map_vp)) * 0.4
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
    plt.savefig(imname, dpi=300)
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
    parser = argparse.ArgumentParser(description='This code creates an attribution map using the integrated gradients method')
    parser.add_argument('-d', "--dataset", type=str, default='amsterdam-umc-db')
    parser.add_argument('-m', "--model", type=str, default='last_state')
    parser.add_argument('-p', "--partition", type=str, default='valid')
    parser.add_argument('-n', "--num_histories", type=int, default=500)
    parser.add_argument('-t', "--truncate", type=int, default=12)
    parser.add_argument('-b', "--num_baselines", type=int, default=32)
    parser.add_argument('-l', "--num_integration_steps", type=int, default=64)
    args = vars(parser.parse_args())
    print('Running with args:', args)

    model = load_pretrained('../results/attribution/%s-%s.pt' % (args['dataset'], args['model']))
    data = pd.read_csv('../../preprocessing/datasets/%s/aggregated_full_cohort_2h/%s.csv' % (args['dataset'], args['partition']))
    features = read_txt('../../preprocessing/datasets/%s/aggregated_full_cohort_2h/state_space_features.txt' % args['dataset'])

    # Which ones to plot (found empirically to be most important across policies/datasets)
    select_only = ['max_vaso_prev', 'total_iv_fluid_prev', 'ventilator', 'sirs_score', 'weight', 'mean_bp', 'dias_bp',
                   'sys_bp', 'aptt', 'lactate', 'creatinine', 'hb', 'hematocrit', 'art_ph']
    select_only_labels = ['Prev. Vasopressors', 'Prev. Fluids', 'Ventilator (on/off)', 'SIRS score', 'Weight', 'Mean BP',
                          'Dias. BP', 'Sys. BP', 'aPTT', 'Lactate', 'Creatinine', 'Hemoglobin', 'Hematocrit', 'Arterial pH']

    # Reproducibility
    random.seed(123)
    torch.manual_seed(123)

    # Augment model with mapping from discrete actions to vasopressor and IV doses
    to_linear_dose = ToLinearDose(ivs=np.array([0, 13.005, 75.885, 151.9, 467.238]) / 500, # median dose of each bin
                                  vps=np.array([0, 0.045, 0.160, 0.342, 0.670]))
    model = torch.nn.Sequential(model, to_linear_dose)

    ig_iv = IntegratedGradients(n_alphas=args['num_integration_steps'], n_baselines=args['num_baselines'], output=0)
    ig_vp = IntegratedGradients(n_alphas=args['num_integration_steps'], n_baselines=args['num_baselines'], output=1)

    attr_maps_iv = []
    attr_maps_vp = []

    for i in tqdm(range(args['num_histories'])):

        # Gather episodes of length 8 (truncate if longer)
        episode = []
        while len(episode) < args['truncate']:
            # Sample episode from dataset
            episode_id = random.choice(data.episode.unique())
            episode = data[data.episode == episode_id].filter(regex='x\d+').values  # only x* columns!
            episode = torch.tensor(episode[-args['truncate']:])

        # Compute attribution map!
        attr_maps_iv.append(ig_iv(episode, model))
        attr_maps_vp.append(ig_vp(episode, model))

    # Compute average map over histories
    attr_map_iv = np.mean(attr_maps_iv, axis=0)
    attr_map_vp = np.mean(attr_maps_vp, axis=0)

    plot(feature_names=features, select_only=select_only, attr_map_iv=attr_map_iv, attr_map_vp=attr_map_vp,
         imname='contribution_map_%s_%s.png' % (args['dataset'], args['model']))
