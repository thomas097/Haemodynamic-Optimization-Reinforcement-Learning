import numpy as np
import pandas as pd
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

# replaces ugly matplot theme
sns.set_theme(style="white")
plt.rcParams.update({'font.size': 32})
plt.rcParams["font.family"] = "Times New Roman"

if __name__ == '__main__':
    # PHWDR scores for 32, 64, 96 and 128 dims as tuples of the form (score, margin)
    phwdr_scores = {
        'AmsterdamUMCdb': {
            'Transformer': [3.054, 3.294, 3.514, 3.434],
            'CKCNN': [3.353, 3.464, 3.430, 3.398]
        },
        'MIMIC-III': {
            'Transformer': [0.207, 2.541, 2.647, 2.384],
            'CKCNN': [1.945, 2.055, 2.549, 2.557]
        }
    }
    # MSE score on validation set with learning objective MT
    valid_mse_scores = {
        'AmsterdamUMCdb': {
            'Transformer': [1.972580575354539734e-01, 1.460090684378222980e-01, 1.105699224789051527e-01, 1.295924582726256935e-01],
            'CKCNN': [2.142054214580043825e-01, 2.082118768809707399e-01, 1.427040022032655675e-01, 1.895334211884030784e-01]
        },
        'MIMIC-III': {
            'Transformer': [1.305783464145139039e-01, 1.367071883272038169e-01, 0.888811618507288109e-01, 1.284355199879336951e-01],
            'CKCNN': [1.673934455521310927e-01, 1.127420562368587325e-01, 1.127040022032655675e-01, 9.156665353582010836e-02]
        }
    }
    dims = [32, 64, 96, 128]

    plt.figure(figsize=(8, 3))

    # MSE + MT
    plt.subplot(1, 2, 1)
    plt.title('Pretraining (MT)')
    for dataset, model_scores in reversed(valid_mse_scores.items()):
        for model, scores in model_scores.items():
            color = 'C0' if model == 'Transformer' else 'C1'
            linestyle = '-' if dataset == 'MIMIC-III' else '--'
            plt.plot(dims, scores, color=color, marker='.', linestyle=linestyle)
    plt.xlabel('Latent dimensions')
    plt.ylabel('MSE')
    plt.grid()

    # PHWDR scores
    plt.subplot(1, 2, 2)
    plt.title('PHWDR')
    for dataset, model_scores in reversed(phwdr_scores.items()):
        for model, scores in model_scores.items():
            color = 'C0' if model == 'Transformer' else 'C1'
            linestyle = '-' if dataset == 'MIMIC-III' else '--'
            plt.plot(dims, scores, label='%s + %s'% (dataset, model), color=color, marker='.', linestyle=linestyle)
    plt.xlabel('Latent dimensions')
    plt.ylabel('Return')
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.show()