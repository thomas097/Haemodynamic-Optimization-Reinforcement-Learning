import pandas as pd
import numpy as np
import torch


class PhysicianEntropy:
    def __init__(self, behavior_policy_file):
        """ Compares the output action probabilities to those by the physician using cross-entropy
        :param behavior_policy_file: Path to DataFrame containing action probabilities (columns '0'-'24') for
                                     behavior policy, chosen actions ('action') and associated rewards ('reward').
        """
        # Extract ations of behavior policy (i.e. physician)
        phys_df = pd.read_csv(behavior_policy_file)
        self._actions = torch.tensor(phys_df.action.values).long()
        self._loss = torch.nn.CrossEntropyLoss()

    def __call__(self, logits):
        """ Computes the cross-entropy loss between the action logits of πe
        and the target actions of πb (the physician's policy)
        :param logits:  Table of action probs acc. to πe with shape (num_states, num_actions)
        :returns:       Cross entropy loss
        """
        return self._loss(torch.tensor(logits), self._actions).item()
