import torch
import numpy as np
import pandas as pd
from baseline_encoders import *
from pretraining import fit_multi_task
from utils import count_parameters, load_pretrained


class MeanEncoder(torch.nn.Module):
    """ Massively underfitting encoder that always predicts the same
    (used for debugging purposes)
    """
    def __init__(self, d):
        super().__init__()
        self._w = torch.nn.Parameter(torch.randn(1, d), requires_grad=True)

    def forward(self, x):
        return self._w.repeat(x.size(0), 1)


if __name__ == '__main__':
    # Training and validation dataset
    train_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_2h/train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_2h/valid.csv')

    in_channels = train_df.filter(regex='x\d+').shape[1]
    out_channels = 32

    # Set up encoder and decoder head
    encoder = MeanEncoder(out_channels) # Replace with encoder of choice, e.g. LastState(in_channels, out_channels)

    # Train!
    fit_multi_task(
        task='mt',
        experiment='results/mean_encoder_pretraining',
        encoder=encoder,
        in_channels=in_channels,
        out_channels=out_channels,
        train_data=train_df,
        valid_data=valid_df,
        mask_missing=True,
        lrate=5e-4,
        epochs=200,
        batches_per_epoch=500,
        warmup=50,
        batch_size=8,
        save_on='valid_mse'
    )
