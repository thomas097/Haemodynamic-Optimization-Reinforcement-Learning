import torch
import numpy as np
import pandas as pd
from baseline_encoders import *
from pretraining import fit_multi_task
from utils import count_parameters, load_pretrained


if __name__ == '__main__':
    # Training and validation dataset
    train_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/valid.csv')

    in_channels = train_df.filter(regex='x\d+').shape[1]
    out_channels = 96

    # Set up encoder and decoder head
    encoder = LastState(in_channels, out_channels)  # Replace with encoder of choice
    print('Encoder parameters:', count_parameters(encoder))


    # Train!
    fit_multi_task(
        task='mt',
        experiment='results/autoencoder_test_pretraining',
        encoder=encoder,
        in_channels=in_channels,
        out_channels=out_channels,
        train_data=train_df,
        valid_data=valid_df,
        mask_missing=True,
        lrate=1e-3,
        epochs=100,
        batches_per_epoch=500,
        warmup=50,
        truncate=256, # much bigger than actual sequence...
        batch_size=8,
        save_on='valid_mse'
    )
