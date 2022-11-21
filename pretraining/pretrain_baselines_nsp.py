import torch
import numpy as np
import pandas as pd
from baseline_encoders import LastState
from next_state_prediction import fit_next_state
from utils import count_parameters


if __name__ == '__main__':
    # Training and validation dataset
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/aggregated_all_1h/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/aggregated_all_1h/mimic-iii_valid.csv')

    # Set up encoder and classification head
    encoder = LastState()
    print('CKCNN parameters:', count_parameters(encoder))

    # decoder to output same number of state-space dims
    decoder = torch.nn.Sequential(
        torch.nn.Linear(48, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 48)
    )

    # Train!
    fit_next_state(
        experiment='results/laststate_nsp_pretraining',
        encoder=encoder,
        decoder=decoder,
        train_data=train_df,
        valid_data=valid_df,
        lrate=1e-4,
        epochs=70,
        batches_per_epoch=1000,
        warmup=25,
        truncate=256, # bigger than actual sequence
        batch_size=8,
        save_on='hubert_loss'
    )