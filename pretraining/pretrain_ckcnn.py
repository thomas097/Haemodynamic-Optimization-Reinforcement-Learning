import torch
import numpy as np
import pandas as pd
from ckcnn import CKCNN
from pretraining import fit_multi_task
from utils import count_parameters


if __name__ == '__main__':
    # Training and validation dataset
    train_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/valid.csv')

    in_channels = train_df.filter(regex='x\d+').shape[1]
    out_channels = 96

    # Set up encoder
    ckcnn = CKCNN(
        layer_channels=(in_channels, 32, 32),
        d_kernel=56,
        max_timesteps=80,
        kernel_type='siren',
        fourier_input=False,
        use_residuals=True,
    )
    ckcnn = torch.nn.Sequential(ckcnn, torch.nn.Linear(32, out_channels)) # final projection
    print('CKCNN parameters:', count_parameters(ckcnn))

    # Train!
    fit_multi_task(
        task='mt',
        experiment='results/amsterdam-umc-db/ckcnn_siren_combined_pretraining_128dims',
        encoder=ckcnn,
        in_channels=in_channels,
        out_channels=out_channels,
        train_data=train_df,
        valid_data=valid_df,
        mask_missing=True,
        lrate=5e-4,
        epochs=200,
        batches_per_epoch=500,
        warmup=50,
        truncate=256,
        batch_size=8,
        save_on='valid_mse'
    )
