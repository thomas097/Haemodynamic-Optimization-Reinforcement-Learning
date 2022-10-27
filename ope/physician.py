import pandas as pd
import numpy as np
import torch


class Physician:
    def __init__(self, behavior_policy_file, drop_terminal_states=True):
        """ Compares the output action probabilities to those by the physician using cross-entropy.

            Params
            behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                  behavior policy, chosen actions ('action') and associated rewards ('reward').
        """
        # Distribution over actions of behavior policy (i.e. physician)
        phys_df = pd.read_csv(behavior_policy_file)
        if drop_terminal_states:
            phys_df = phys_df[phys_df['reward'].notna()]
        self._pi_b = self._behavior_policy(phys_df)
        self._loss = torch.nn.CrossEntropyLoss()

    @staticmethod
    def _behavior_policy(df):
        # Return numeric columns (assumed to belong to actions, i.e., '0' - '24')
        action_cols = sorted([c for c in df.columns if c.isnumeric()], key=lambda c: int(c))
        return df[action_cols].values.astype(np.float64)

    def __call__(self, pi_e):
        """ Computes the cross-entropy loss between the action probabilities of πe
            and the target policy πb (the physician's policy).

            pi_e:       Table of action probs acc. to πe with shape (num_states, num_actions)

            Returns:    Estimate of mean V^πe
        """
        # Set impossible actions with pi_e values of zero to one (prevent log(0))
        pi_e[pi_e < 1e-6] = 1
        ln = -np.sum(self._pi_b * np.log(pi_e), axis=1)
        return np.mean(ln)
