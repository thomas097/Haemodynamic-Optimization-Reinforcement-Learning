import os
import torch
import pathlib as pl
import pandas as pd


def load_pretrained(path):
    """ Loads pretrained model from path """
    if not os.path.exists(path):
        raise Exception('File %s does not exist' % path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    model.eval()
    model.zero_grad()
    return model


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
    print('episodes: %d' % len(df.episode.unique()))
    print('memory: %.1fGB\n' % (df.memory_usage(deep=True).sum() / (1 << 27)))
    return df


def count_parameters(model):
    """ Computes the number of learnable parameters """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
