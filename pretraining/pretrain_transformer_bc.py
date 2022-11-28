import torch
import numpy as np
import pandas as pd
from vanilla_transformer import Transformer
from behavior_cloning import fit_behavior_cloning
from utils import count_parameters


if __name__ == '__main__':
    # Training and validation dataset
    train_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h/train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h/valid.csv')

    # Drop IV/VP as they force a trivial solution
    train_df = train_df.drop(['x0', 'x1'], axis=1)
    valid_df = valid_df.drop(['x0', 'x1'], axis=1)

    # Set up encoder and classification head
    transformer = Transformer(
        in_channels=53,
        out_channels=32,
        d_model=32,
        n_heads=4,
        n_blocks=2,
        d_key=16,
        causal=True
    )
    print('Transformer parameters:', count_parameters(transformer))

    classifier = torch.nn.Sequential(
        torch.nn.Linear(32, 96),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(96, 25),
    )

    # Train!
    fit_behavior_cloning(
        experiment='results/transformer_bc_pretraining',
        encoder=transformer,
        classifier=classifier,
        train_data=train_df,
        valid_data=valid_df,
        oversample_vaso=0,
        batches_per_epoch=500,
        epochs=200,
        warmup=25,
        lrate=1e-3,
        batch_size=32,
        truncate=256,  # way big
        save_on=False
    )
