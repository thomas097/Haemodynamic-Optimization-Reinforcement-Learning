import os
import torch
import torch.nn.functional as F
import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from tqdm import tqdm
from types import MethodType


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


def sample_episode(dataset, episode_id=None, time_point=None, maxlen=256):
    """ Samples a history from a random time point in a random episode
    :param dataset:     DataFrame containing validation dataset
    :param episode_id:  Optional episode id (randomly selected from dataset otherwise)
    :param time_point:  Optional time point (randomly selected from dataset otherwise)
    :param maxlen:      Maximum number of observations in the past
    :returns:           FloatTensor of shape (1, n_observations, n_features)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # sample episode
    if episode_id is None:
        episode_id = random.choice(dataset.episode.unique())
    episode = dataset[dataset.episode == episode_id]

    # sample history up to some random time point from episode
    if time_point is None:
        time_point = random.choice(episode[episode.action.notna()].index)
    history = episode[episode.index <= time_point].filter(regex='x\d+')

    # truncate observations to maxlen
    history = history.values[-maxlen:]
    return torch.Tensor(history).unsqueeze(0).to(device)


def monkey_forward(self, x, x_encoded, return_last=True):
    """ Forcibly modifies forward method of Transformer using 'monkey patching' to insert
    a pre-one-hot encoded input x_encoded to allow us to compute the gradient of the input
    x w.r.t. an output channel (see compute_gradient())
    :param self:         Reference to instance
    :param x:            Standard input to model (See forward())
    :param x_encoded:    Pre-computed one-hot encoded input equiv. to x of shape (batch_size, n_observations, n_channels + value)
                         against which to compute the gradient w.r.t the output
    :param return_last:  Whether to return the output of the final timestep only
    :return:             Tensor of shape (batch_size, num_timesteps, out_channels)
                         or (batch_size, out_channels) when `return_last=True`
    """
    f, v, t = torch.split(x, 1, dim=2)

    # Create combined causal/padding/modality mask
    src_mask = self._padding_mask(x)
    if self._causal:
        causal_mask = self._causal_mask(t)
        src_mask = torch.logical_or(src_mask, causal_mask).detach()

    if self._mask_features:
        feature_mask = self._feature_mask(f)
        src_mask = torch.logical_or(src_mask, feature_mask).detach()

    # Compute RPE distances
    rel_dist = self._distance_matrix(t).detach()

    y = self._input_encoding(x_encoded) # inject x_encoded straight into embedding network (bypassing one-hot)!
    for layer in self._encoder_layers:
        y = layer(y, src_mask=src_mask, rel_dists=rel_dist)

    # only mask padding, but allow 'fusion' of channels
    y = self._fusion(y, src_mask=self._padding_mask(x), rel_dists=rel_dist)

    return self._linear(y[:, -1] if return_last else y)


def encode_input(x, vocab_size):
    """ Encodes sequence of (feature, value, charttime) pairs into a latent vector
    :param x:  Tensor of shape (batch_size, num_timesteps, 3) where the features
               correspond to measurement_type, value and timestep respectively
    :returns:  Tensor of shape (batch_size, num_timesteps, d_model)
    """
    f, v, _ = torch.split(x, 1, dim=2)

    # One-hot encode features (i.e. measurement types) and jointly model with values
    f = F.one_hot(f[:, :, 0].long(), num_classes=vocab_size)
    return torch.concat([f, v], dim=2)


def integrated_gradient(x, x_encoded, model, out_channel, steps=32):
    """ Computes attribution scores for each observation w.r.t. some output
    state-space feature using Integrated Gradients (Sundararajan et al., 2017)
    For details, see: https://arxiv.org/abs/1703.01365
    """
    # compute linear interpolations between x_encoded and a baseline (all zeros)
    alphas = torch.linspace(0.0, 1.0, steps).unsqueeze(1).unsqueeze(1)
    z_encoded = alphas * x_encoded

    # compute gradients of z_encoded w.r.t. output
    z_encoded.requires_grad = True
    model.zero_grad()
    loss = -model(x, z_encoded)[0, out_channel]
    loss.backward()
    grads = z_encoded.grad.detach().numpy()

    # Riemann's approximation of integral
    out = np.mean((grads[:-1] + grads[1:]) / 2, axis=0)

    # Only return gradients w.r.t. value of observation!
    return out[:, -1]


def plot_attribution(grads, inputs, caption, feature_names):
    """ Plots attribution as a scatter plot where color intensity marks importance
    :param grads:         ndarray of gradients w.r.t to the input
    :param inputs:        ndarray of inputs of shape (n_observations, 3) containing
                          the feature, value and measurement time, respectively
    :param feature_names: Names of the features in column 0 of `inputs`
    """
    sns.set_theme(style="darkgrid")
    sns.set_style("whitegrid")

    # shift decision point to t=0
    inputs[:, 2] -= np.max(inputs[:, 2])

    # Drop attribution to 'action' (it naturally receives most attribution)
    action_id = feature_names.index('action')
    feature_names.pop(action_id)
    grads = grads[inputs[:, 0] != action_id + 1]
    inputs = inputs[inputs[:, 0] != action_id + 1]

    # plot measurements with attribution as increasing intensity
    fig, ax = plt.subplots()
    sc = ax.scatter(inputs[:, 2], inputs[:, 0], c=grads, cmap='bwr', edgecolors='grey')
    fig.colorbar(sc, ax=ax)
    ax.set_title(caption)
    ax.set_yticks(range(1, len(feature_names) + 1))
    ax.set_yticklabels(feature_names)
    ax.set_xlabel('Time (h)')
    ax.set_ylabel('Feature')
    plt.show()


if __name__ == '__main__':
    model = load_pretrained("../results/transformer_v2_pretraining_00000/encoder.pt")
    dataset = pd.read_csv('../../preprocessing/datasets/mimic-iii/non_aggregated_1h/mimic-iii_valid.csv')
    features = pd.read_csv('../../preprocessing/datasets/mimic-iii/non_aggregated_1h/measurement_ids.txt', header=None)[0]

    # Monkey patch forward method of transformer!
    model.forward = MethodType(monkey_forward, model)

    inputs = sample_episode(dataset, episode_id=223357, time_point=190359, maxlen=256)
    encoded = encode_input(inputs, vocab_size=46)

    out_channels = model.config['out_channels']
    for channel in range(out_channels):
        plot_attribution(
            grads=integrated_gradient(inputs, encoded, model, out_channel=channel),
            inputs=inputs[0].numpy(),
            caption='Output channel (%d/%d)' % (channel + 1, out_channels),
            feature_names=features.tolist()
        )
