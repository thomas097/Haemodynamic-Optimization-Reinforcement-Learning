import numpy as np
import pandas as pd
from ckcnn import CKCNN
from pretraining import fit_behavior_cloning
from utils import count_parameters


if __name__ == '__main__':
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv')
    print('train_df.size = %d  valid_df.size = %d\n' % (len(train_df), len(valid_df)))

    encoder = CKCNN(layer_channels=(48, 64), kernel_dims=24, max_timesteps=18)
    print('CKCNN parameters: %d\n' % count_parameters(encoder))

    fit_behavior_cloning(experiment_name='results/ckcnn_pretraining',
                         encoder=encoder,
                         classif_layer_shapes=(64, 48),
                         num_actions=25,
                         train_dataset=train_df,
                         valid_dataset=valid_df,
                         oversample_vaso=5,  # To account for small number of vaso treatments in training set
                         lrate=1e-3,
                         epochs=100,
                         batch_size=8,
                         eval_after=1,
                         save_on_best=True)