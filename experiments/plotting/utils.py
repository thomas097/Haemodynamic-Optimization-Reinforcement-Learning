import os
import io
import pickle
import torch
import numpy as np
import pandas as pd
import pathlib as pl
from tqdm import tqdm
from experience_replay import EvaluationReplay


def load_actions_to_bins(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def load_txt(path):
    with open(path, 'r') as file:
        return [l.strip() for l in file.readlines()]


def load_pretrained(path):
    """ Loads pretrained model from path """
    if not os.path.exists(path):
        raise Exception('File %s does not exist' % path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    model.eval()
    model.zero_grad()
    return model


def evaluate_policy_on_dataset(encoder, policy, dataset, _type='qvals'):
    # Load dataset into replay buffer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay = EvaluationReplay(dataset, return_history=encoder is not None, device=device, max_len=256)

    # Feed histories through encoder to get fixed state representation
    with torch.no_grad():
        if encoder is not None:
            encoded_states = torch.concat([encoder(t).detach() for t in tqdm(replay.iterate(1))])
        else:
            encoded_states = torch.concat([t for t in replay.iterate()])

    # Return Q-values according to model
    if _type == 'qvals':
        return policy(encoded_states).detach().numpy()
    elif _type == 'actions':
        return policy.sample(encoded_states)
    elif _type == 'action_probs':
        return policy.action_probs(encoded_states)
    else:
        raise Exception("_type %s not understood" % _type)


def run_encoder_over_dataset(encoder, dataset):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay = EvaluationReplay(dataset, return_history=True, device=device, max_len=256)
    return torch.concat([encoder(t).detach() for t in replay.iterate()]).detach().numpy()


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