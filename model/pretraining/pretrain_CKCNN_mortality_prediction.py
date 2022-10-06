import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from ckconv import CKConv


class CKCNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, num_classes=2):
        super().__init__()
        self._conv1 = CKConv(in_channels, hidden_channels)
        self._conv2 = CKConv(hidden_channels, hidden_channels)
        self._linear = torch.nn.Linear(hidden_channels, num_classes)
        self._activation = torch.nn.ELU()

    def forward(self, x, predict=True):
        h1 = self._activation(self._conv1(x))
        y = self._activation(self._conv2(h1)[:, -1])  # Feed final representation into dense layer
        if predict:
            return torch.log_softmax(self._linear(y), dim=1)
        else:
            return y

    def encode(self, x):
        return self.forward(x, predict=False)

    def fit(self, X, y, epochs=100, batch_size=8, lrate=1e-3, eval_after=100):
        criterion = torch.nn.NLLLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lrate)

        for ep in tqdm(range(epochs)):
            # Shuffle samples (xi, yi)
            indices = torch.randperm(X.shape[0])
            X, y = X[indices], y[indices]

            total_loss = 0
            for i in range(0, X.shape[0], batch_size):
                X_batch = X[i:i + batch_size]
                y_batch = y[i:i + batch_size]

                loss = criterion(self(X_batch), y_batch)
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if ep % eval_after == 0:
                print('ep %s: loss = %.2f' % (ep, total_loss))


def load_dataset(path, state_features, reward_col, episode_col):
    df_train = pd.read_csv(path, index_col=0, usecols=state_features + [reward_col, episode_col])

    # Create matrix of the form (num_icu_stays, num_states, num_features)
    X, y = [], []
    for _, group_df in df_train.groupby(episode_col):
        X.append(group_df[state_features].values)
        y.append((group_df[reward_col].values[-2] + 15) / 30)

    # Return as Tensors
    X = torch.Tensor(np.array(X))
    y = torch.LongTensor(np.array(y))
    return X, y


if __name__ == '__main__':
    # Define columns marking state- and action-space
    STATE_SPACE_FEATURES = ['max_vp', 'total_iv_fluid', 'sirs_score', 'sofa_score', 'weight', 'ventilator', 'height',
                            'age', 'gender', 'heart_rate', 'temp', 'mean_bp', 'dias_bp', 'sys_bp', 'resp_rate', 'spo2',
                            'natrium', 'chloride', 'kalium', 'trombo', 'leu', 'anion_gap', 'aptt', 'art_ph', 'asat',
                            'alat', 'bicarbonaat', 'art_be', 'ion_ca', 'lactate', 'paco2', 'pao2', 'hb', 'bilirubin',
                            'creatinine', 'inr', 'ureum', 'albumin', 'magnesium', 'calcium', 'glucose',
                            'total_urine_output']
    REWARD_COL = 'reward'
    EPISODE_COL = 'icustay_id'

    # Load preprocessed dataset
    X, y = load_dataset('../../preprocessing/datasets/mimic-iii/handcrafted/mimic-iii_train_handcrafted.csv',
                        STATE_SPACE_FEATURES, REWARD_COL, EPISODE_COL)
    print('X.shape = %s  y.shape = %s' % (X.shape, y.shape))

    # Fit CKCNN to predict final rewards (scaled to death=0 and survival=1)
    ckcnn = CKCNN(in_channels=len(STATE_SPACE_FEATURES), hidden_channels=32, num_classes=2)
    ckcnn.fit(X, y, epochs=100, batch_size=16, lrate=1e-5, eval_after=10)
