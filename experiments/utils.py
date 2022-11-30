import os
import torch
import pathlib as pl
import numpy as np
import pandas as pd
from tqdm import tqdm
from experience_replay import EvaluationReplay
from physician_entropy import PhysicianEntropy
from per_horizon_importance_sampling import PHWIS


class OPECallback:
    def __init__(self, behavior_policy_file, valid_data, batch_size=128):
        """ Callback to evaluate policy during training
        :param behavior_policy_file:  Estimated behavior policy
        :param valid_data:            Validation set
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._replay = EvaluationReplay(valid_data, device=device)
        self._phwis = PHWIS(behavior_policy_file)
        self._phys_ent = PhysicianEntropy(behavior_policy_file)
        self._batch_size = batch_size

    def _softmax(self, x, temp=2):
        z = np.exp(x / temp)
        return z / np.sum(z, axis=1, keepdims=True)

    def __call__(self, model):
        """ Evaluated encoder-policy pair using Weighted Importance Sampling on validation set
        :param encoder:  History encoding model
        """
        with torch.no_grad():
            action_logits = []
            with tqdm(desc='evaluating', position=0, leave=True) as pbar:
                for states in self._replay.iterate(self._batch_size):
                    # histories -> encoder -> policy network + softmax
                    logits = model(states).cpu().detach().numpy()
                    action_logits.append(logits)
                    pbar.update(states.size(0))

        # convert torch tensors to numpy ndarray
        action_logits = np.concatenate(action_logits, axis=0)

        # metrics
        phwis = self._phwis(self._softmax(action_logits))
        ess = self._phwis.ess
        phys_entropy = self._phys_ent(action_logits)
        return {'phwis': phwis, 'ess': ess, 'physician_entropy': phys_entropy}


def load_data(path, add_missingness=False):
    """ Utility function to load in dataset with minimal memory footprint """
    # Load dataset and cast to efficient datatypes
    df = pd.read_csv(path)
    df['episode'] = df.episode.astype('category')
    df['action'] = df.action.astype('float16')  # not uint as can contain NaNs :(
    df['reward'] = df.reward.astype('float16')  # not int as can contain NaNs
    for i in df.filter(regex='x\d+').columns:
        df[i] = df[i].astype('float32')

    # Print current memory usage
    print('%s:' % pl.Path(path).stem)
    print('size:  ', df.shape)
    print('memory: %.1fGB\n' % (df.memory_usage(deep=True).sum() / (1 << 27)))
    return df


def load_pretrained(path):
    """ Loads pretrained model from path """
    if not os.path.exists(path):
        raise Exception('File %s does not exist' % path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    model.eval()
    model.zero_grad()
    return model


def count_parameters(model):
    """ Computes the number of learnable parameters """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)