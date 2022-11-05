import os
import torch


def load_pretrained(path):
    """ Loads pretrained model from path """
    if not os.path.exists(path):
        raise Exception('File %s does not exist' % path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.load(path, map_location=device)
