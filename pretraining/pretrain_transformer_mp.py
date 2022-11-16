import torch
import numpy as np
import pandas as pd
from transformer_models import Transformer
from mortality_prediction import fit_mortality_prediction
from utils import count_parameters, load_data


if __name__ == '__main__':
    # Training and validation dataset
    train_df = load_data('../preprocessing/datasets/mimic-iii/non_aggregated_1h/mimic-iii_train.csv')
    valid_df = load_data('../preprocessing/datasets/mimic-iii/non_aggregated_1h/mimic-iii_valid.csv')

    # Set up encoder and classification head
    transformer = Transformer(vocab_size=46, out_channels=32, d_model=32, n_heads=4, n_blocks=2, d_key=16)
    print('Transformer parameters:', count_parameters(transformer))

    classifier = torch.nn.Sequential(
        torch.nn.Linear(32, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 1),
        torch.nn.Sigmoid()
    )

    # Train!
    fit_mortality_prediction(experiment='results/transformer_v2_pretraining',
                             encoder=transformer,
                             classifier=classifier,
                             train_data=train_df,
                             valid_data=valid_df,
                             lrate=1e-4,
                             epochs=100,
                             batches_per_epoch=1000,
                             class_weights=(1, 3),
                             warmup=25,
                             truncate=128,
                             batch_size=8,
                             save_on='valid_loss')
