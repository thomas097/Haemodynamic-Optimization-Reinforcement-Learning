import os
import torch
import pathlib as pl
import pandas as pd
from experience_replay import EvaluationReplay
from importance_sampling import WeightedIS
from physician_entropy import PhysicianEntropy


class Callback:
    def __init__(self, behavior_policy_file, valid_data):
        """ Callback to evaluate policy during training
        :param behavior_policy_file:  Estimated behavior policy
        :param valid_data:            Validation set
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._wis = WeightedIS(behavior_policy_file)
        self._phys = PhysicianEntropy(behavior_policy_file)
        self._replay = EvaluationReplay(valid_data, return_history=True, device=device)

    def __call__(self, encoder, policy, batch_size=128):
        with torch.no_grad():
            encoded_states = torch.concat([encoder(t).detach() for t in self._replay.iterate(batch_size)])
            action_probs = policy.action_probs(encoded_states)

        weighted_is = self._wis(action_probs)
        phys_entropy = self._phys(action_probs)
        return {'wis': weighted_is, 'physician_entropy': phys_entropy}


def load_data(path):
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