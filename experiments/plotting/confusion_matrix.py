import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from experience_replay import EvaluationReplay


def to_numpy(x):
    return x.cpu().detach().numpy()


def load_pretrained(path):
    """ Load pretrained pytorch model from file
    :param path:  Path to Transformer instance in pt format
    :returns:     A PyTorch model
    """
    if not os.path.exists(path):
        raise Exception('%s does not exist' % path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(path, map_location=device)
    model.eval()
    return model


def get_actions(model, dataset_file, batch_size):
    """ Obtains actions for each history in dataset from policy
    :param model:       A trained policy network
    :param dataset:     Dataset with variable-length patient trajectories
    :param batch_size:  Number of histories to process at once
    :returns:           Tensor of shape (n_states, n_actions)
    """
    # load dataset into replay buffer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = pd.read_csv(dataset_file)
    replay = EvaluationReplay(dataset=dataset, device=device)

    with torch.no_grad():
        actions_pred = []
        with tqdm(total=len(dataset), desc='gathering predictions', position=0, leave=True) as pbar:
            for states in replay.iterate(batch_size):
                # histories -> encoder -> policy network + argmax
                probs = to_numpy(torch.argmax(model(states), dim=1))
                actions_pred.append(probs)
                pbar.update(states.size(0))

    # convert torch tensors to numpy ndarray
    actions_pred = np.concatenate(actions_pred, axis=0)
    actions_true = dataset.action.values
    return actions_true, actions_pred


def plot_confusion_matrix(model, dataset_file, maxlen, batch_size):
    """ Plots confusion table of model predictions """
    # Collect predictions of model
    model.eval()
    y_true, y_pred = get_actions(
        model=model,
        dataset_file=dataset_file,
        batch_size=batch_size,
    )

    conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(25), normalize='true')
    print(classification_report(y_true, y_pred))

    plt.matshow(conf_mat)
    plt.title('Confusion table')
    plt.xlabel('Predicted action')
    plt.ylabel('Physician\'s action')
    plt.show()


if __name__ == '__main__':
    model = load_pretrained("../results/pretrained_transformer_experiment_00002/model.pt")
    dataset_file = "../../preprocessing/datasets/amsterdam-umc-db_v2/aggregated_full_cohort_2h/test.csv"

    plot_confusion_matrix(
        model=model,
        dataset_file=dataset_file,
        maxlen=256,
        batch_size=8,
    )
