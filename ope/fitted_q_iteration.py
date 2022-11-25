import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from fitted_q_evaluation import FittedQEvaluation


class FittedQIteration(FittedQEvaluation):
    """ Implementation of the Fitted Q-Iteration (FQI) estimator for Off-policy
        Policy Evaluation (OPE). For details, see:
        http://proceedings.mlr.press/v139/hao21b/hao21b.pdf
    """
    def __init__(self, training_file, num_actions=25, gamma=1.0, lrate=1e-2, iters=1000, early_stopping=50):
        super(FittedQIteration, self).__init__(
            training_file=training_file,
            num_actions=num_actions,
            gamma=gamma,
            lrate=lrate,
            iters=iters,
            early_stopping=early_stopping
        )

    def fit(self, *args, **kwargs):
        """
        Fits estimator for Q^πe to states, actions, rewards and next states obtained through
        a behavior policy πb. Note: In FQI, the behavior policy need not be supplied.
        """
        # Mask out expected future reward at next state if terminal
        reward_mask = (self._train.rewards == 0).float()

        # perform policy iteration
        avg_q_vals = []
        with tqdm(range(self._iters)) as pbar:
            no_improvement = 0

            for _ in pbar:
                # Q-estimate
                q_pred = self._estimator(self._train.states, hard_actions=self._train.actions)

                # Bootstrapped target
                with torch.no_grad():
                    exp_future_reward = torch.max(self._estimator(self._train.next_states), axis=1)[0]
                    q_next = self._train.rewards + self._gamma * reward_mask * exp_future_reward

                # Update!
                self._optimizer.zero_grad()
                self._criterion(q_pred, q_next).backward()
                self._optimizer.step()

                # print average Q-value to screen
                avg_q = torch.mean(q_pred).item()
                pbar.set_postfix({'avg_q': avg_q})
                avg_q_vals.append(avg_q)

                # early stopping (seek improvement of at least 1e-2 above x episodes ago)
                if avg_q > avg_q_vals[-self._early_stopping:][0] + 1e-2:
                    no_improvement = 0
                elif no_improvement == self._early_stopping:
                    break
                else:
                    no_improvement += 1

        self._fitted = True
        return self

    def state_value(self, policy_action_probs):
        """
        Returns the estimated state value, V(s), according to the
        evaluation policy on which the FQE instance was fitted.
        """
        if not self._fitted:
            raise Exception('Estimator has not been fitted; Call fit().')

        policy_actions = torch.argmax(torch.Tensor(policy_action_probs), dim=1)
        return self._estimator(self._train.all_states, hard_actions=policy_actions).detach().numpy()

    def state_action_value(self):
        """
        Returns the estimated state-action value, Q(s, a), according to the
        evaluation policy on which the FQE instance was fitted.
        """
        if not self._fitted:
            raise Exception('Estimator has not been fitted; Call fit().')

        return self._estimator(self._train.all_states).detach().numpy()


if __name__ == '__main__':
    # Behavior policy
    behavior_df = pd.read_csv('physician_policy/amsterdam-umc-db_aggregated_full_cohort_1h_knn/valid_behavior_policy.csv')
    behavior_action_probs = behavior_df.filter(regex='\d+').values  # assume 25 actions

    # Zero policy
    zerodrug_action_probs = np.zeros(behavior_action_probs.shape)
    zerodrug_action_probs[:, 0] = 1

    # Random policy
    np.random.seed(42)
    random_action_probs = np.random.uniform(0, 1, behavior_action_probs.shape)
    random_action_probs = random_action_probs / np.sum(random_action_probs, axis=1, keepdims=True)

    # Fit FQEs
    # Note: always use aggregated data file for training (time points in non-aggregated file will match!)
    training_file = '../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h/valid.csv'
    behavior_estimator_fqi = FittedQIteration(training_file).fit()
    print('Behavior: ', np.mean(behavior_estimator_fqi.state_value(behavior_action_probs)))

    zerodrug_estimator_fqi = FittedQIteration(training_file).fit()
    print('Zero-drug:', np.mean(zerodrug_estimator_fqi.state_value(zerodrug_action_probs)))

    random_estimator_fqi = FittedQIteration(training_file).fit()
    print('Random:   ', np.mean(random_estimator_fqi.state_value(random_action_probs)))

