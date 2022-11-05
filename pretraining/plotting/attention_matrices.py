import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_pretrained
from pretraining import DataLoader

sns.set_theme(style="darkgrid")
sns.set_style("whitegrid", {'axes.grid': False})


def plot_attention(model, data_df, add_leading_states=1):
    """ Plots attention matrices for inputs in data_df """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(data_df, device=device)

    for x, _ in dataloader.iterate(batch_size=1):

        # Duplicates state 0 N additional times to simulate longer trajectory
        if add_leading_states > 0:
            leader = x[:, :1].repeat(1, int(add_leading_states), 1)
            leader = leader + 0.05 * torch.randn(leader.shape)  # Little bit of Gaussian noise
            x = torch.concat([leader, x], dim=1)

        matrices = model.get_attention_matrices(x)

        num_blocks = matrices.shape[0]

        plt.Figure(figsize=(12, 4))
        for i in range(num_blocks):
            plt.subplot(1, num_blocks, i + 1)
            plt.imshow(matrices[i, 0])
            plt.title('Layer %d' % i)
            plt.xlabel('Keys')
            plt.ylabel('Queries')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    model = load_pretrained("../results/transformer_pretraining_00039/encoder.pt")

    data_df = pd.read_csv("../../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv")

    plot_attention(model, data_df, add_leading_states=10)
