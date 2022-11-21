import torch
import numpy as np
import pandas as pd
from vanilla_transformer import Transformer
from behavior_cloning import fit_behavior_cloning
from utils import count_parameters, load_data


if __name__ == '__main__':
    # Training and validation dataset
    train_df = load_data('../preprocessing/datasets/mimic-iii/aggregated_1h/mimic-iii_train.csv')
    valid_df = load_data('../preprocessing/datasets/mimic-iii/aggregated_1h/mimic-iii_valid.csv')

    train_df = train_df.drop(['x0', 'x1'], axis=1)
    valid_df = valid_df.drop(['x0', 'x1'], axis=1)

    # Set up encoder and classification head
    transformer = Transformer(
        in_channels=46,
        out_channels=32,
        d_model=32,
        n_heads=4,
        n_blocks=3,
        d_key=16,
        causal=True
    )
    print('Transformer parameters:', count_parameters(transformer))

    classifier = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 25)
    )

    # Train!
    fit_behavior_cloning(
        experiment='results/transformer_bc_pretraining',
        encoder=transformer,
        classifier=classifier,
        train_data=train_df,
        valid_data=valid_df,
        lrate=1e-4,
        epochs=100,
        oversample_vaso=6000,  # Add additional trajectories with infrequent actions by oversampling
        batches_per_epoch=1000,
        warmup=25,
        truncate=256, # bigger than actual sequence
        batch_size=8,
        save_on='valid_loss'
    )
