import torch
import pandas as pd


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
    return ce(q_vals, true_actions[:, 0])


def conservative_regularizer(q_all, q_chosen_actions):
    """ Applies a conservative Q-learning regularizer to minimize over-estimation of
    Out-Of-Distribution (OOD) actions (underlying catastrophic collapse during training)
    For details, see:
    (Kumar et al., 2020) CQL: https://arxiv.org/pdf/2006.04779.pdf
    OR
    (Kaushik et al., 2022) CQL for sepsis: https://arxiv.org/pdf/2203.13884.pdf
    """
    return torch.mean(torch.log(torch.sum(torch.exp(q_all), dim=1))) - torch.mean(q_chosen_actions)


class ProbRegularizer:
    def __init__(self, behavior_policy_file, thres=0.2):
        """ Regularizer to punish the policy when deviating above some threshold
        from the behavior policy's action probabilities
        :param behavior_policy_file:  File containing action probabilities of physician estimated from dataset
        :param thres:                 Maximum difference pi_b - pi_e
        """
        self._thres = thres

        # extract action probabilities of actions chosen by behavior policy
        behavior_policy = pd.read_csv(behavior_policy_file)
        action_probs = torch.tensor(behavior_policy.filter(regex='\d+').values)
        self._actions = torch.tensor(behavior_policy.action.values).long().unsqueeze(1)
        self._action_probs = torch.gather(
            input=action_probs,
            dim=1,
            index=self._actions
        )

    def __call__(self, qvals, state_indices):
        # determine action probs acc. to model
        pi_e = torch.gather(
            input=torch.softmax(qvals, dim=1),
            dim=1,
            index=self._actions[state_indices]
        )

        # punish policy if probs differ above some threshold
        pi_b = self._action_probs[state_indices]
        loss = torch.clamp((pi_b - pi_e) - self._thres, min=0)
        return torch.mean(loss)


if __name__ == '__main__':
    LCBRegularizer('../../ope/physician_policy/aggregated_all_1h/mimic-iii_train_behavior_policy.csv')


