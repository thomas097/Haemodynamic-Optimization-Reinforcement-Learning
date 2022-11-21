import torch
import numpy as np
import pandas as pd
from baseline_encoders import LastState
from behavior_cloning import fit_behavior_cloning
from utils import count_parameters, load_data


if __name__ == '__main__':
    # Training and validation dataset
    train_df = load_data('../preprocessing/datasets/mimic-iii/aggregated_1h/mimic-iii_train.csv')
    valid_df = load_data('../preprocessing/datasets/mimic-iii/aggregated_1h/mimic-iii_valid.csv')

    # # prevent over-reliance on previous IV/VP doses by dropping these features
    train_df = train_df.drop(['x0', 'x1'], axis=1)
    valid_df = valid_df.drop(['x0', 'x1'], axis=1)

    # set up encoder and classification head
    lstm = LastState()
    print('LSTM parameters:', count_parameters(lstm))

    classifier = torch.nn.Sequential(
        torch.nn.Linear(46, 64),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(64, 25)
    )

    # train!
    fit_behavior_cloning(
        experiment='results/laststate_bc_pretraining',
        encoder=lstm,
        classifier=classifier,
        train_data=train_df,
        valid_data=valid_df,
        lrate=1e-3,
        epochs=70,
        oversample_vaso=6000,  # Add additional trajectories with infrequent actions by oversampling
        batches_per_epoch=1000,
        warmup=25,
        truncate=256, # bigger than actual sequence so no truncation at all
        batch_size=8,
        save_on='valid_loss'
    )
