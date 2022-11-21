import torch
import numpy as np
import pandas as pd
from vanilla_transformer import Transformer
from next_state_prediction import fit_next_state
from utils import count_parameters


if __name__ == '__main__':
    # Training and validation dataset
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/aggregated_all_1h/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/aggregated_all_1h/mimic-iii_valid.csv')

    # Set up encoder and classification head
    transformer = Transformer(
        in_channels=48,
        out_channels=32,
        d_model=32,
        n_heads=4,
        n_blocks=2,
        d_key=16,
        causal=True
    )
    print('Transformer parameters:', count_parameters(transformer))

    # decoder to output same number of state-space dims
    decoder = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 48)
    )

    # Train!
    fit_next_state(
        experiment='results/transformer_nsp_pretraining',
        encoder=transformer,
        decoder=decoder,
        train_data=train_df,
        valid_data=valid_df,
        lrate=1e-4,
        epochs=100,
        batches_per_epoch=1000,
        warmup=50,
        truncate=256, # bigger than actual sequence
        batch_size=8,
        save_on='hubert_loss'
    )
