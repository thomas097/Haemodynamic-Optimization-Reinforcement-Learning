import torch
import matplotlib.pyplot as plt
from utils import load_pretrained


def plot_kernels(convnet, in_channels, out_channel, positions):
    """ Plots kernels as a function of artificial inputs
    :param convnet:      Trained convolutional network
    :param in_channels:  Number of input channels / features
    :param positions:    Positions at which to generate data
    """
    # prepare artificial dataset with all features at all positions
    X = []
    for in_channel in range(in_channels):
        x = torch.zeros(1, positions.size(0), 3)
        x[0, :, 0] = torch.full((positions.size(0),), fill_value=in_channel)
        x[0, :, 2] = positions
        X.append(x)
    X = torch.concat(X, dim=1)

    # obtain kernels for input
    kernels = []
    for kernel in convnet.kernels(X):
        if len(kernel.shape) == 4:  # AsyncCKConv
            kernel = torch.reshape(kernel[0, out_channel, -1], (-1, in_channels)).t()
        else:
            kernel = kernel[out_channel]  # regular CKConv / causal conv
        kernels.append(kernel.detach().numpy().transpose())

    # plot kernels side by side
    plt.figure(figsize=(12, 8))
    for layer, kernel in enumerate(kernels):
        plt.subplot(1, 2, layer + 1)
        plt.imshow(kernel)
        plt.title('layer %d' % (layer + 1))
        plt.xlabel('In channel')
        plt.ylabel('Time')
    plt.show()


if __name__ == '__main__':
    # Load model from file
    #ckcnn = load_pretrained('roggeveen_experiment_blah_blah')
    from async_ckcnn import AsyncCKCNN

    timesteps = torch.linspace(1, 140, 101)  # todo: remove
    ckcnn = AsyncCKCNN(
        in_channels=47,
        hidden_channels=56,
        out_channels=96,
        d_kernel=16,
        positions=timesteps
    )

    plot_kernels(convnet=ckcnn,
                 in_channels=46,
                 out_channel=3,
                 positions=torch.linspace(0, 72, 101))
