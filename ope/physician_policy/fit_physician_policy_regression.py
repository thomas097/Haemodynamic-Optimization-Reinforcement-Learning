import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path, PurePath


class Policy(torch.nn.Module):
    """ Implementation of multinomial softmax regression using PyTorch """
    def __init__(self, input_dim, output_dim, lrate=1e-3, batch_size=128, epochs=20):
        super().__init__()
        self._linear = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64, 96),
            torch.nn.ELU(),
            torch.nn.Linear(96, output_dim)
        )
        self._lrate = lrate
        self._batch_size = batch_size
        self._epochs = epochs

    def forward(self, X):
        return self._linear(X)

    def fit(self, X, y):
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self._linear.parameters(), lr=self._lrate)

        X = torch.tensor(X).float()
        y = torch.tensor(y).long().flatten()

        print('Training...')
        for ep in range(self._epochs):
            total_loss = []

            for i in range(0, X.size(0), self._batch_size):
                X_batch = X[i: i + self._batch_size]
                y_batch = y[i: i + self._batch_size]

                loss = criterion(self(X_batch), y_batch)
                total_loss.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print('Ep %d: loss = %.2f' % (ep, np.mean(total_loss)))

        print('- Done!')

    def predict_proba(self, X):
        x = torch.tensor(X).float()
        logits = self(x)
        return torch.softmax(self(x), dim=1).detach().numpy()


def estimate_behavior_policy(train_set, batch_size=128):
    """ LinearRegression-based behavior policy estimation for OPE.
    Please refer to (Raghu et al., 2018) for details.
    train_set:  Path to preprocessed training set with states and chosen actions (by physician)
    returns:    Fitted LinearRegression model
    """
    # Load training set into DataFrame
    train_df = pd.read_csv(train_set)
    states = train_df.filter(regex='x\d+').values
    actions = train_df.action.values

    print('\nFitting policy to %s states...' % states.shape[0])

    lr = Policy(input_dim=states.shape[1], output_dim=25, batch_size=batch_size)
    lr.fit(states, actions)
    return lr


def onehot(x, num_classes):
    x = torch.tensor(x).long()
    onehot = torch.nn.functional.one_hot(x, num_classes=num_classes)
    return onehot.detach().numpy()


def evaluate_policy(policy, dataset, batch_size=128):
    """ Applies policy to held-out dataset and estimates distribution over actions.

        policy:        kNN model fitted to physician actions
        dataset:       Dataset to evaluate policy on
        batch_size:    Size of batches used for evaluation (default: 128)
    """
    # Load train/valid/test set into DataFrame
    df = pd.read_csv(dataset)
    episode = df['episode'].values
    actions = df['action'].values
    rewards = df['reward'].values
    states = df.filter(regex='x\d+').values
    indices = df.index.values

    action_probs = []

    # Process batch-wise to reduce overload
    print('\nProcessing %s\n' % dataset)
    for i in tqdm(range(0, len(indices), batch_size)):
        index_batch = indices[i: i + batch_size]

        # Look up actions of k-nearest states in training set
        batch_action_probs = policy.predict_proba(states[index_batch])

        # Interpolate with chosen action by 5% to ensure a non-zero probability
        # is given to the action chosen by the physician
        batch_action_onehot = onehot(actions[index_batch], num_classes=25)
        batch_action_probs = 0.95 * batch_action_probs + 0.05 * batch_action_onehot

        action_probs.append(batch_action_probs)

    # Stack batches
    action_probs = np.concatenate(action_probs, axis=0)

    # Combine action probs with chosen actions and rewards
    res = pd.DataFrame(data=np.column_stack([episode, actions, rewards, action_probs]),
                       index=indices, columns=['episode', 'action', 'reward'] + list(range(25)))
    return res


if __name__ == '__main__':
    DATASET = '../../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h'

    #######################
    #   Estimate Policy   #
    #######################

    # Step 0. create output directory, e.g. mimic-iii-aggregated_1h
    path = PurePath(DATASET)
    out_dir = '%s_%s_lr' % (path.parent.name, path.name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Step 1. estimate behavior policy from training set
    train_file = os.path.join(DATASET, 'train.csv')
    policy = estimate_behavior_policy(train_file)

    # Step 2. compute action distribution of behavior policy over each dataset
    for part in ['train', 'valid', 'test']:
        data_file = os.path.join(DATASET, part + '.csv')
        action_probs = evaluate_policy(policy, data_file)

        outfile = os.path.join(out_dir, '%s_behavior_policy.csv' % part)
        action_probs.to_csv(outfile, index=False)

    # Sanity Check: is estimated policy predictive of the true policy's actions?
    from sklearn.metrics import f1_score, accuracy_score

    valid_file = os.path.join(DATASET, 'valid.csv')
    valid_behavior_policy_file = os.path.join(out_dir, 'valid_behavior_policy.csv')

    true_actions = pd.read_csv(valid_file).action
    action_probs = pd.read_csv(valid_behavior_policy_file).filter(regex='\d+').values
    predicted_actions = np.argmax(action_probs, axis=1)

    print('Predictive of its own actions?')
    print('F1: ', f1_score(true_actions, predicted_actions, average='macro'))
    print('Acc:', accuracy_score(true_actions, predicted_actions))






