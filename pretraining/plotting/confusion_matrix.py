import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from dataloader import DataLoader
from utils import load_pretrained, to_numpy


def soft_confusion_matrix(y_true, y_pred, labels, normalize):
    out = np.zeros((len(labels), len(labels)), dtype=np.float64)
    for i, j in zip(y_true, y_pred):
        out[i] += j
    if normalize:
        out = out / np.sum(out, axis=1, keepdims=True)
    return out


def predict_on_dataset(model, data_df, maxlen=256, batch_size=8, seed=42):
    """ Collects predictions of model on dataset """
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(data_df, device=device, maxlen=maxlen)

    # No need for backprop ;)
    with torch.no_grad():
        y_pred, y_true = [], []

        with tqdm(desc='Collecting predictions') as pbar:
            for states, actions in tqdm(dataloader.iterate(batch_size)):
                # Predict action class using model
                model_pred = model(states)
                action_preds = torch.argmax(model_pred, dim=1)
                y_pred.append(to_numpy(action_preds))
                y_true.append(to_numpy(actions))
                pbar.update(batch_size)

    # Concatenate collected predictions as vector
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return y_true, y_pred


def plot_confusion_matrix(model, data_df, maxlen, batch_size):
    """ Plots confusion table of model predictions """
    # Collect predictions of model
    model.eval()
    y_true, y_pred = predict_on_dataset(
        model=model,
        data_df=data_df,
        maxlen=maxlen,
        batch_size=batch_size,
    )

    conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(25), normalize='true')

    plt.matshow(conf_mat)
    plt.title('Confusion table')
    plt.xlabel('Predicted action')
    plt.ylabel('Physician\'s action')
    plt.show()

    # Print per-class precision, recall and F1
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    model = load_pretrained("../results/transformer_v2_pretraining_00002/classifier.pt")
    data = pd.read_csv("../../preprocessing/datasets/mimic-iii/aggregated_1h/mimic-iii_valid.csv")

    # prevent over-reliance on previous IV/VP doses by dropping these features
    # data = data.drop(['x0', 'x1'], axis=1)

    plot_confusion_matrix(
        model=model,
        data_df=data,
        maxlen=256,
        batch_size=8,
    )
