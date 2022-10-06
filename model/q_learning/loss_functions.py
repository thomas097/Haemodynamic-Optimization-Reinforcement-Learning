import torch


def weighted_Huber_loss(x, y, weights, delta=1.0):
    """ HuberLoss with per-sample (importance) weights """
    a = 0.5 * torch.pow(x - y, 2)
    b = delta * (torch.abs(x - y) - 0.5 * delta)
    mask = (torch.absolute(x - y) < delta).float()
    loss = mask * a + (1 - mask) * b
    return torch.mean(weights * loss)


def reward_regularizer(q_pred, max_reward):
    """ Punishes policy for overestimating Q-values above max_reward """
    return torch.clamp(torch.abs(q_pred) - max_reward, min=0).sum().double()
