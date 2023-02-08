# to allow running from terminal without pyCharm :(
import sys

sys.path.append('../ope')
sys.path.append('../models/q_learning')
sys.path.append('../models/attention/nn')
sys.path.append('../models/ckconv/nn')
sys.path.append('../models')

import torch
import argparse
import numpy as np
import pandas as pd
from ckcnn import CKCNN
from pretraining import fit_multi_task
from utils import count_parameters


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trains ckcnn encoder on desired pretraining task')
    parser.add_argument('-d', "--dataset", type=str, default='amsterdam-umc-db')
    parser.add_argument('-t', "--task", type=str, default='mt')
    parser.add_argument('-o', "--out_dims", type=int, default=96)
    parser.add_argument('-l', "--lrate", type=float, default=5e-4)
    parser.add_argument('-e', "--epochs", type=int, default=200)
    parser.add_argument('-z', "--batches_per_epoch", type=int, default=500)
    parser.add_argument('-w', "--warmup", type=int, default=50)
    parser.add_argument('-b', "--batch_size", type=int, default=32)
    args = vars(parser.parse_args())
    print('Running with args:', args)

    # Training and validation dataset
    train_df = pd.read_csv('../preprocessing/datasets/%s/aggregated_full_cohort_2h/train.csv' % args['dataset'])
    valid_df = pd.read_csv('../preprocessing/datasets/%s/aggregated_full_cohort_2h/valid.csv' % args['dataset'])

    in_channels = train_df.filter(regex='x\d+').shape[1]
    out_channels = args['out_dims']

    # Set up encoder
    ckcnn = CKCNN(
        layer_channels=(in_channels, 32, 32),
        d_kernel=56,
        max_timesteps=128,
        kernel_type='siren',
        fourier_input=False,
        use_residuals=True,
    )
    ckcnn = torch.nn.Sequential(ckcnn, torch.nn.Linear(32, out_channels)) # final projection
    print('CKCNN parameters:', count_parameters(ckcnn))

    # Train!
    fit_multi_task(
        task=args['task'],
        experiment='results/ckcnn_%s_pretraining' % args['task'],
        encoder=ckcnn,
        in_channels=in_channels,
        out_channels=out_channels,
        train_data=train_df,
        valid_data=valid_df,
        mask_missing=True,
        lrate=args['lrate'],
        epochs=args['epochs'],
        batches_per_epoch=args['batches_per_epoch'],
        warmup=args['warmup'],
        truncate=256,
        batch_size=args['batch_size'],
        save_on='valid_mse'
    )
