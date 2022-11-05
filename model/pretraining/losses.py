import torch
import torch.nn.functional as F
import numpy as np


def _one_hot(y_true, mask, num_classes):
    y_true = y_true.masked_fill(~mask, 0)  # Temporarily encode masked entries as class 0
    return F.one_hot(y_true, num_classes).float()


def cross_entropy(y_pred, y_true, mask, num_classes, device='gpu'):
    """ Masked CrossEntropy loss for 3D tensors with shape (B, L, C)
    """
    target = _one_hot(y_true, mask, num_classes).to(device)
    log_pred = torch.log_softmax(y_pred, dim=2)

    loss = -torch.sum(target * log_pred, dim=2)
    loss = loss.masked_fill(~mask, 0)
    return torch.sum(loss) / (torch.sum(mask) + 1)





