import torch
import numpy as np
import pandas as pd
from transformer_models import CausalTransformer
from pretraining import fit_behavior_cloning
from utils import count_parameters, load_data


if __name__ == '__main__':
    # Training and validation dataset
    train_df = load_data('../preprocessing/datasets/mimic-iii/attention_4h/mimic-iii_train.csv')
    valid_df = load_data('../preprocessing/datasets/mimic-iii/attention_4h/mimic-iii_valid.csv')

    # Set up encoder and classifier head
    transformer = CausalTransformer(vocab_size=46, out_channels=96, d_model=64, n_heads=4, n_blocks=2, d_key=16)
    print('Transformer parameters:', count_parameters(transformer))

    classifier = torch.nn.Sequential(
        torch.nn.Linear(96, 96),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(96, 25)
    )

    # Train!
    fit_behavior_cloning(experiment='results/transformer_v2_pretraining',
                         encoder=transformer,
                         classifier=classifier,
                         train_data=train_df,
                         valid_data=valid_df,
                         lrate=1e-3,
                         epochs=100,
                         batches_per_epoch=1000,
                         truncate=256,
                         batch_size=8,
                         save_on='valid_loss')
