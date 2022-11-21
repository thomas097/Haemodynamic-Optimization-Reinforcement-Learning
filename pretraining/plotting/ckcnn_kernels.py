import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


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


def plot_kernels(convnet, out_channel, layer, feature_names):
    """ Plots kernels as a function of artificial inputs
    :param convnet:        Trained (continuous kernel) convolutional network
    :param out_channel:    Output channel / feature for which to visualize kernel
    :param layer:          Layer at which to visualize kernels
    :param feature_names:  List of input feature names (no known for hidden layers)
    """
    # select kernel of specific out channel at specified layer
    kernel = convnet.kernels[layer][out_channel]

    # color map where the highest magnitude sets min/max of color range
    rng = np.max(np.abs(kernel))
    divnorm = colors.TwoSlopeNorm(vmin=-rng, vcenter=0., vmax=rng)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)
    ax.imshow(kernel, cmap='bwr', norm=divnorm)

    # labels
    n_features, n_timesteps = kernel.shape
    ax.set_xlabel('Time step')
    ax.set_ylabel('Input channel')
    ax.set_title('Convolution kernel at layer %d of output channel %d' % (layer, out_channel))
    ax.yaxis.tick_right()
    ax.set_yticks(np.arange(n_features))
    if feature_names is not None:
        ax.set_yticklabels(feature_names)
    else:
        ax.set_yticklabels(['h%d' % f for f in range(n_features)])
    ax.yaxis.set_ticks_position('right')

    # grid lines
    ax.set_xticks(np.arange(-.5, n_timesteps, 1), minor=True)
    ax.set_yticks(np.arange(-.5, n_features, 1), minor=True)
    ax.grid(which='minor', color='w', linestyle='-', linewidth=1)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # Load model from file
    model = load_pretrained("../results/ckcnn_nsp_relu_pretraining_00001/encoder.pt")
    features = read_txt('../../preprocessing/datasets/mimic-iii/aggregated_all_1h/state_space_features.txt')

    for layer in range(1):
        for out_channel in range(16):
            plot_kernels(
                convnet=model,
                out_channel=out_channel,
                layer=layer,
                feature_names=features if layer == 0 else None,
            )
