import torch
import numpy as np
import pandas as pd
from vanilla_transformer import Transformer
from pretraining import fit_multi_task
from utils import count_parameters, load_pretrained


if __name__ == '__main__':
    # Training and validation dataset
    train_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/valid.csv')

    in_channels = train_df.filter(regex='x\d+').shape[1]
    out_channels = 96

    # Set up encoder
    transformer = Transformer(
        in_channels=in_channels,
        out_channels=out_channels,
        d_model=64,
        n_heads=4,
        n_blocks=2,
        d_key=12,
        conv_size=3,
    )
    print('Transformer parameters:', count_parameters(transformer))

    # Train!
    fit_multi_task(
        task='mt',
        experiment='results/amsterdam-umc-db/transformer_combined_pretraining_128dims',
        encoder=transformer,
        in_channels=in_channels,
        out_channels=out_channels,
        train_data=train_df,
        valid_data=valid_df,
        mask_missing=True,
        lrate=5e-4,
        epochs=300,
        batches_per_epoch=500,
        warmup=50,
        truncate=256,
        batch_size=8,
        save_on='valid_mse'
    )
