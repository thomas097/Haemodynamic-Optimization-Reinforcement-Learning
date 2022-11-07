import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm
from sklearn.metrics import confusion_matrix, classification_report
from dataloader import DataLoader
from utils import load_pretrained, to_numpy


def predict_on_dataset(model, data_df, maxlen=256, batch_size=8, num_samples=200, seed=42):
    """ Collects predictions of model on dataset """
    np.random.seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = DataLoader(data_df, device=device, maxlen=maxlen)

    # No need for backprop ;)
    with torch.no_grad():
        y_pred, y_true = [], []
        total_samples = 0

        with tqdm(desc='Collecting predictions') as pbar:
            for states, actions in dataloader.iterate(batch_size):
                # Predict action class using model
                action_preds = torch.argmax(model(states), dim=1)
                y_pred.append(to_numpy(action_preds))
                y_true.append(to_numpy(actions))
                pbar.update(batch_size)

                total_samples += batch_size
                if total_samples > num_samples:
                    break

    # Concatenate collected predictions as vector
    y_true = np.concatenate(y_true, axis=0)
    y_pred = np.concatenate(y_pred, axis=0)
    return y_true, y_pred


def plot_confusion_matrix(model, data_df, maxlen, batch_size, num_samples):
    """ Plots confusion table of model predictions """
    # Collect predictions of model
    model.eval()
    y_true, y_pred = predict_on_dataset(model, data_df, maxlen=maxlen, batch_size=batch_size, num_samples=num_samples)

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
    model = load_pretrained("../results/transformer_v2_pretraining_00000/classifier (8).pt")
    data = pd.read_csv("../../preprocessing/datasets/mimic-iii/attention_2h/mimic-iii_valid.csv")

    plot_confusion_matrix(model=model,
                          data_df=data,
                          maxlen=256,
                          batch_size=8,
                          num_samples=5000)
