import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from pretraining import DataLoader
from utils import load_pretrained, to_numpy


def predict_on_dataset(model, data_df, truncate=512, seed=42):
    """ Collects predictions of model on dataset """
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(data_df, device=device)

    # No need for backprop ;)
    with torch.no_grad():
        y_pred, y_true = [], []

        with tqdm(desc='Predicting...') as pbar:
            for k, (x, y) in enumerate(dataloader.iterate(batch_size=1)):
                # Truncate excessively long sequences
                if x.shape[1] > truncate:
                    i = np.random.randint(0, x.shape[1] - truncate)
                    x = x[:, i:i + truncate]
                    y = y[:, i:i + truncate]

                # Predict class using model
                p = torch.argmax(model(x), dim=2)

                # Mask predictions for -1 class (looking at you Transformer)
                mask = to_numpy(y >= 0).flatten()
                y = to_numpy(y).flatten()[mask]
                p = to_numpy(p).flatten()[mask]

                y_pred.append(p)
                y_true.append(y)
                pbar.update(1)

                if k == 500:
                    break

    # Concatenate collected predictions as vector
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return y_true, y_pred


def plot_confusion_matrix(model, data_df):
    """ Plots confusion table of model predictions """
    # Collect predictions of model
    y_true, y_pred = predict_on_dataset(model, data_df)

    # Compute confusion matrix
    conf_mat = confusion_matrix(y_true, y_pred, labels=np.arange(25), normalize='true')

    plt.matshow(conf_mat)
    plt.title('Confusion table')
    plt.xlabel('Predicted action')
    plt.ylabel('Physician\'s action')
    plt.show()

    # Print per-class precision, recall and F1
    print(classification_report(y_true, y_pred))


if __name__ == '__main__':
    model = load_pretrained("../results/transformer_v2_pretraining_00000/classifier.pt")
    data_df = pd.read_csv("../../preprocessing/datasets/mimic-iii/attention_4h/mimic-iii_valid.csv")
    plot_confusion_matrix(model, data_df)