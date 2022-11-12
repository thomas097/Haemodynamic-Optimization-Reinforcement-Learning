import torch
import numpy as np
import pandas as pd
from async_ckcnn import AsyncCKCNN
from behavior_cloning import fit_behavior_cloning
from utils import count_parameters, load_data


if __name__ == '__main__':
    # Training and validation dataset
    train_df = load_data('../preprocessing/datasets/mimic-iii/non_aggregated_4h/mimic-iii_train.csv')
    valid_df = load_data('../preprocessing/datasets/mimic-iii/non_aggregated_4h/mimic-iii_valid.csv')

    # Set up encoder and classification head
    ckconv = AsyncCKCNN(
        in_channels=47,
        hidden_channels=56,
        out_channels=96,
        d_kernel=16,
        positions=torch.linspace(1, 72, 71)
    )
    print('CKConv parameters:', count_parameters(ckconv))

    classifier = torch.nn.Sequential(
        torch.nn.Linear(96, 96),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(96, 25)
    )

    # Train!
    fit_behavior_cloning(experiment='results/async_ckconv_pretraining',
                         encoder=ckconv,
                         classifier=classifier,
                         train_data=train_df,
                         valid_data=valid_df,
                         lrate=1e-3,
                         epochs=100,
                         oversample_vaso=6000,   # Add additional trajectories with infrequent actions by oversampling
                         batches_per_epoch=100,
                         truncate=256,
                         batch_size=8,
                         save_on='valid_loss')
