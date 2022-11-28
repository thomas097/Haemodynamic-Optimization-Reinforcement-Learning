import torch
import numpy as np
import pandas as pd
from ckcnn import CKCNN
from next_state_prediction import fit_next_state
from utils import count_parameters


if __name__ == '__main__':
    # Training and validation dataset
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/aggregated_all_1h/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/aggregated_all_1h/mimic-iii_valid.csv')

    # Set up encoder and next-state-prediction head
    encoder = CKCNN(
        layer_channels=(48, 32),
        d_kernel=24,
        max_timesteps=72,
        kernel_type='siren',
        fourier_input=True,
        use_residuals=True,
    )
    print('CKCNN parameters:', count_parameters(encoder))

    # decoder to output same number of state-space dims
    decoder = torch.nn.Sequential(
        torch.nn.Linear(32, 32),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(32, 48),
    )

    # Train!
    fit_next_state(
        experiment='results/ckcnn_siren_nsp_pretraining',
        encoder=encoder,
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
