import torch
import numpy as np
import pandas as pd
from baseline_encoders import *
from next_state_prediction import fit_next_state
from utils import count_parameters


if __name__ == '__main__':
    # Training and validation dataset
    train_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h/train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h/valid.csv')

    # Set up encoder and next-state regression decoder
    encoder = LSTM()
    print('Encoder parameters:', count_parameters(encoder))

    # decoder to output same number of state-space dims
    decoder = torch.nn.Sequential(
        torch.nn.Linear(55, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 55)
    )

    # Train!
    fit_next_state(
        experiment='results/lstm_nsp_pretraining',
        encoder=encoder,
        decoder=decoder,
        train_data=train_df,
        valid_data=valid_df,
        lrate=5e-4,
        epochs=100,
        batches_per_epoch=500,
        warmup=25,
        truncate=8,
        batch_size=8,
        save_on='valid_mse'
    )
