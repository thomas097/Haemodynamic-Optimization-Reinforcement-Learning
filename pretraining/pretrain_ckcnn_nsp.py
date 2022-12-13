import torch
import numpy as np
import pandas as pd
from ckcnn import CKCNN
from next_state_prediction import fit_next_state
from utils import count_parameters


if __name__ == '__main__':
    # Training and validation dataset
    train_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/valid.csv')

    in_channels = train_df.filter(regex='x\d+').shape[1]

    # Set up encoder and next-action prediction head
    ckcnn = CKCNN(
        layer_channels=(in_channels, 32),
        d_kernel=24,
        max_timesteps=72,
        kernel_type='siren',
        fourier_input=False,
        use_residuals=True,
    )
    print('CKCNN parameters:', count_parameters(ckcnn))

    # decoder to output same number of state-space dims
    decoder = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, in_channels),
    )

    # Train!
    fit_next_state(
        experiment='results/ckcnn_siren_nsp_pretraining',
        encoder=ckcnn,
        decoder=decoder,
        train_data=train_df,
        valid_data=valid_df,
        lrate=5e-4,
        epochs=100,
        batches_per_epoch=1000,
        warmup=25,
        truncate=256, # bigger than actual sequence
        batch_size=8,
        save_on='valid_mse'
    )
