import numpy as np
import pandas as pd
from experimental_transformer import CausalTransformer
from pretraining import fit_behavior_cloning
from utils import count_parameters


if __name__ == '__main__':
    # Training and validation dataset
    train_df = pd.read_csv('../../preprocessing/datasets/mimic-iii/attention_4h/mimic-iii_train.csv')
    valid_df = pd.read_csv('../../preprocessing/datasets/mimic-iii/attention_4h/mimic-iii_valid.csv')
    print('train_df.size = %d  valid_df.size = %d' % (len(train_df), len(valid_df)))

    # Set up model
    transformer = CausalTransformer(vocab_size=46, out_channels=96, d_model=64, d_key=32, n_blocks=3, truncate=50)
    print('Transformer parameters:', count_parameters(transformer))

    fit_behavior_cloning(experiment_name='results/transformer_v2_pretraining',
                         encoder=transformer,
                         classif_layer_shapes=(96, 64),
                         num_actions=25,
                         train_dataset=train_df,
                         valid_dataset=valid_df,
                         oversample_vaso=5,  # To account for small number of vaso treatments in training set
                         lrate=1e-4,
                         epochs=100,
                         batch_size=32,
                         eval_after=1,
                         save_on_best=True)
