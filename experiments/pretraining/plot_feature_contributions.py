import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_pretrained
from pretraining import DataLoader

from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
)

sns.set_theme(style="darkgrid")
sns.set_style("whitegrid", {'axes.grid': False})

torch.manual_seed(123)
np.random.seed(123)


def extract_output_timestep(model, x, timestep=-1):
    return model(x)[:, timestep]


def plot_feature_contributions(model, data_df, timestep=-1, feature=0):
    """ Plots input feature contributions to output features """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(data_df, device=device)

    # The output must be 1D, so we pick the state representation of a single timestep
    model_at_timestep = lambda x: extract_output_timestep(model, x, timestep=timestep)

    for x, y in dataloader.iterate(batch_size=1):
        actions = y[0][y[0] >= 0].cpu().detach().numpy()
        num_actions = len(actions)

        # Compute feature importance using Captum library
        baseline = torch.zeros(*x.shape)
        ig = IntegratedGradients(model_at_timestep)
        attributions, delta = ig.attribute(x, baseline, target=feature, return_convergence_delta=True)

        plt.figure(figsize=(12, 6))
        plt.subplot(2, 1, 1)
        plt.imshow(attributions[0].numpy().T, interpolation='nearest', aspect='auto')
        plt.title('Attribution scores')
        plt.ylabel('Input features')
        plt.xlabel('Timesteps')
        plt.xticks(ticks=np.arange(num_actions), labels=np.arange(num_actions))
        plt.xlim(-0.5, num_actions - 0.5)

        plt.subplot(2, 1, 2)
        plt.plot(torch.arange(num_actions), actions)
        plt.title('Actions')
        plt.xlabel('Timesteps')
        plt.ylabel('Action')
        plt.xticks(ticks=np.arange(num_actions), labels=np.arange(num_actions))
        plt.xlim(-0.5, num_actions - 0.5)

        plt.tight_layout()
        plt.show()

        print('Convergence Delta:', delta.item())


if __name__ == '__main__':
    model = load_pretrained("../results/transformer_pretraining_00039/encoder.pt")

    data_df = pd.read_csv("../../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv")

    plot_feature_contributions(model, data_df, feature=0)
