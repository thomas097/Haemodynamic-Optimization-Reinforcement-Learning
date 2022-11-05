import numpy as np
import pandas as pd
from transformer_models_v2 import CausalTransformer
from pretraining import fit_behavior_cloning
from utils import count_parameters, load_data


if __name__ == '__main__':
    # Training and validation dataset
    train_df = load_data('../preprocessing/datasets/mimic-iii/attention_4h/mimic-iii_train.csv')
    valid_df = load_data('../preprocessing/datasets/mimic-iii/attention_4h/mimic-iii_valid.csv')

    # Set up model
    transformer = CausalTransformer(vocab_size=46, out_channels=96, d_model=64, n_heads=3, d_head=16, d_key=16, truncate=0)
    print('Transformer parameters:', count_parameters(transformer))

    fit_behavior_cloning(experiment_name='results/transformer_v2_pretraining',
                         encoder=transformer,
                         classif_layer_shapes=(96, 64),
                         num_actions=25,
                         train_dataset=train_df,
                         valid_dataset=valid_df,
                         oversample_vaso=True,  # To account for small number of vaso treatments in training set
                         lrate=1e-4,
                         epochs=100,
                         truncate=100,
                         batch_size=32,
                         eval_after=1,
                         save_on_best=True)
