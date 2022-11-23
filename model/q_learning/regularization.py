import torch
import pandas as pd


class ESSRegularization:
    def __init__(self, behavior_policy_file, n_timesteps=56, n_rollouts=100, min_sample_size=50):
        """ Regularizes the model by enforcing a minimum effective sample size (ESS) in the WIS
        estimator. This way we obtain a policy that can be evaluated using OPE, while not forcing
        the policy to follow the behavior policy exactly.
        :param behavior_policy_file:  File containing actions and probabilities under behavior policy
        :param n_timesteps:           Number of time steps (i.e. the horizon of the RL environment)
        :param n_rollouts:            Number of virtual, or 'pseudo', episodes to roll out
        :param min_sample_size:       Minimum sample size (min_sample_size <= n_virtual_episodes)
        """
        # extract action and action probabilities of behavior policy ove training set
        behavior_policy = pd.read_csv(behavior_policy_file)
        actions = behavior_policy.action.values
        probas = behavior_policy.filter(regex=r'\d+').values

        # cast to PyTorch tensors
        self._actions = torch.tensor(actions).long().unsqueeze(1)
        self._probas = torch.tensor(probas)

        # horizon of WIS, i.e. number of time steps in episode
        self._n_timesteps = n_timesteps
        self._n_rollouts = n_rollouts
        self._min_sample_size = min_sample_size

    @staticmethod
    def _gather(probas, index):
        return torch.gather(input=probas, dim=1, index=index)

    def __call__(self, q_values, state_indices):
        """ Computes a loss inversely proportional to the sample size of the WIS estimator
        :param q_values:       Tensor of Q-values of shape (n_states, n_actions)
        :param state_indices:  LongTensor of indices of training set states associated with Q-values
        :returns:              Loss value inverse proportional to sample size
        """
        # compute probabilities of actions chosen by physician under physician- and evaluation-policies
        pi_b = self._gather(self._probas[state_indices], index=self._actions[state_indices])
        pi_e = self._gather(torch.softmax(q_values, dim=1), index=self._actions[state_indices])

        # compute point-wise importance ratio
        ratio = pi_e / pi_b

        # approximate cumulative importance weights at terminal time step through random re-sampling of ratios
        n_samples = ratio.size(0)
        idx = torch.randint(low=0, high=n_samples, size=(self._n_rollouts, self._n_timesteps))
        weights = torch.prod(ratio[idx], dim=1)

        # compute effective sample size
        norm_weights = weights / torch.sum(weights)
        ess = 1 / torch.sum(torch.pow(norm_weights, 2))

        # maximize sample size if below threshold (then allow exploratory freedom)
        return torch.clamp(self._min_sample_size - ess, min=0)


if __name__ == '__main__':
    reg = ESSRegularization('behavior_policy_file.csv')
    loss = reg(torch.randn(8, 25), torch.randint(0, 100, (8,)))
    print(loss)
