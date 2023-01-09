import os
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.decomposition import IncrementalPCA, FastICA
from sklearn.manifold import TSNE
from experience_replay import EvaluationReplay

# replaces ugly matplot theme
plt.rcParams.update({'font.size': 14})
plt.rcParams["font.family"] = "Times New Roman"


def load_pretrained(path):
    """ Load pretrained pytorch model from file """
    if not os.path.exists(path):
        raise Exception('%s does not exist' % path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    model.eval()
    return model


def load_csv(path):
    """ Load CSV file using pandas """
    if not os.path.exists(path):
        raise Exception('%s does not exist' % path)
    return pd.read_csv(path).reset_index(drop=True)


def load_pickle(path):
    """ Load pickled file """
    if not os.path.exists(path):
        raise Exception('%s does not exist' % path)
    with open(path, 'rb') as file:
        return pickle.load(file)


def encode(model, dataset, truncate=256, batch_size=8):
    """ determines proportion of IV/VP actions (0 - 4) chosen by model over time
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay_buffer = EvaluationReplay(dataset, device=device, max_len=truncate)

    # Collect encodings
    encodings = []
    with torch.no_grad():
        with tqdm(total=dataset.shape[0]) as pbar:
            for x in replay_buffer.iterate(batch_size):
                # feed batch of states (x) through model to predict action
                encodings.append(model(x, return_last=True).detach().numpy())
                pbar.update(x.size(0))
    return np.concatenate(encodings, axis=0)



if __name__ == '__main__':
    dataset = load_csv('../../preprocessing/datasets/amsterdam-umc-db_v3/aggregated_full_cohort_2h/valid.csv')

    # use encoder to encode trajectories in dataset
    # encodings = encode(
    #     model=load_pretrained('../results/amsterdam-umc-db/transformer_experiment_00000/encoder.pt'),
    #     dataset=dataset
    # )
    encodings = dataset.filter(regex='x\d+').values # Handcrafted State

    # extract states near end of trajectory to speed up t-SNE
    idx = dataset.groupby('episode', as_index=False).tail(6).index
    encodings = encodings[idx]

    # reduce dimensionality to two using t-SNE
    encodings_small = TSNE(
        n_components=3,
        verbose=2,
        perplexity=10,
        n_iter=2000,
        random_state=2
    ).fit_transform(encodings)
    print('Finished t-SNE')

    # assign color to different outcomes
    def apply_outcome(r):
        terminal_r = r.values[-1]
        return r * 0 + terminal_r
    colors = dataset.reward.groupby(dataset.episode).apply(apply_outcome).values
    colors = colors[idx]

    # Plot
    ax = plt.figure().add_subplot(projection='3d')

    ax.scatter(encodings_small[:, 0], encodings_small[:, 1], zs=encodings_small[:, 2],
               c=colors, zdir='y', alpha=0.5, cmap='viridis')

    ax.set_xlabel('Dim 0')
    ax.set_ylabel('Dim 1')
    ax.set_zlabel('Dim 2')
    ax.set_xlim(np.min(encodings_small[:, 0]), np.max(encodings_small[:, 0]))
    ax.set_ylim(np.min(encodings_small[:, 1]), np.max(encodings_small[:, 1]))
    ax.set_zlim(np.min(encodings_small[:, 2]), np.max(encodings_small[:, 2]))

    ax.view_init(elev=19., azim=-30)
    plt.savefig('tsne_new_extra.png', dpi=300)
    plt.tight_layout()
    plt.show()


