import torch
import numpy as np
import pandas as pd
from ckcnn import CKCNN
from behavior_cloning import fit_behavior_cloning
from utils import count_parameters


if __name__ == '__main__':
    # Training and validation dataset
    train_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h/train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h/valid.csv')

    # Drop IV/VP as they force a trivial solution
    train_df = train_df.drop(['x0', 'x1'], axis=1)
    valid_df = valid_df.drop(['x0', 'x1'], axis=1)

    # Set up encoder and next-action prediction head
    encoder = CKCNN(
        layer_channels=(53, 32),
        d_kernel=24,
        max_timesteps=72,
        kernel_type='siren',
        fourier_input=False,
        use_residuals=True,
    )
    print('CKCNN parameters:', count_parameters(encoder))

    classifier = torch.nn.Sequential(
        torch.nn.Linear(32, 96),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(96, 25),
    )

    # Train!
    fit_behavior_cloning(
        experiment='results/ckcnn_siren_bc_pretraining',
        encoder=encoder,
        classifier=classifier,
        train_data=train_df,
        valid_data=valid_df,
        oversample_vaso=0,
        epochs=200,
        warmup=25,
        lrate=1e-3,
        batch_size=32,
        truncate=256, # way big
        save_on=False
    )
