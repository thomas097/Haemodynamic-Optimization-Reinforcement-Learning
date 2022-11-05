import os
import pickle
import matplotlib.pyplot as plt


def get_model(model_path):
    """ Loads encoder model stored in `model_path`
    """
    # Load encoder
    encoder_path = os.path.join(model_path, 'encoder.pkl')
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as file:
            return pickle.load(file)
    else:
        raise Exception('No `encoder.pkl` in directory %s' % model_path)


def plot_kernel(kernels, layer=1, in_channel=1, out_channel=1):
    weights = kernels[layer][out_channel][in_channel]
    plt.plot(weights)
    plt.title('layer=%s out_channel=%s in_channel=%s' % (layer, out_channel, in_channel))
    plt.xlabel('Timestep')
    plt.show()


if __name__ == '__main__':
    # Load model from file
    ckcnn = get_model('roggeveen_experiment_blah_blah')

    # Get kernels from CKCNN layers
    kernels = ckcnn.kernels
    print(kernels.shape)
