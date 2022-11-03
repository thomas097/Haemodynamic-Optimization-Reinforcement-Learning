import numpy as np
import pandas as pd
from transformer_models import CausalTransformer
from pretraining import fit_behavior_cloning


if __name__ == '__main__':
    # Training and validation dataset
    train_df = pd.read_csv('../preprocessing/datasets/mimic-iii/attention_4h/mimic-iii_train.csv')
    valid_df = pd.read_csv('../preprocessing/datasets/mimic-iii/attention_4h/mimic-iii_valid.csv')

    # Set up model
    transformer = CausalTransformer(vocab=np.arange(46), out_channels=96, d_model=64, d_key=16, n_blocks=3, truncate=0)

    fit_behavior_cloning(experiment_name='transformer_pretraining',
                         encoder=transformer,
                         classif_layer_sizes=(96, 64, 25),  # 25 actions!
                         dataset=train_df,
                         callback=None,
                         lrate=1e-3,
                         epochs=100,
                         batch_size=16,
                         eval_after=1,
                         save_on_best=True)