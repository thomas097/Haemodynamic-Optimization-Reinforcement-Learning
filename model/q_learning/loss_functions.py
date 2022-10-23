import torch


def weighted_Huber_loss(x, y, weights, delta=1.0):
    """ HuberLoss with per-sample (importance) weights """
    a = 0.5 * torch.pow(x - y, 2)
    b = delta * (torch.abs(x - y) - 0.5 * delta)
    mask = (torch.absolute(x - y) < delta).float()
    loss = mask * a + (1 - mask) * b
    return torch.mean(weights * loss)


def weighted_MSE_loss(x, y, weights):
    """ HuberLoss with per-sample (importance) weights """
    mse_loss = torch.pow(x - y, 2)
    return torch.mean(weights * mse_loss)


def reward_regularizer(q_pred, limit):
    """ Punishes policy for overestimating Q-values above or below limit """
    return torch.clamp(torch.abs(q_pred) - limit, min=0).sum().double()


def physician_regularizer(q_vals, true_actions):
    """ Force networks action probabilities to lie closely to those of the physician. """
    ce = torch.nn.CrossEntropyLoss()
    pred_action_probs = torch.softmax(q_vals, dim=1)
    return ce(pred_action_probs, true_actions[:, 0])
