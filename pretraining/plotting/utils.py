import os
import torch


def load_pretrained(path):
    """ Loads pretrained model from file """
    if not os.path.exists(path):
        raise Exception('%s does not exist' % path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    model.eval()
    return model


def to_numpy(tensor):
    """ Converts tensor (possibly on GPU) to Numpy ndarray """
    return tensor.detach().cpu().numpy()