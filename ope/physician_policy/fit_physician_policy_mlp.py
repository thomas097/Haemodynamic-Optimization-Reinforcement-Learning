import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
from pathlib import Path, PurePath


class Policy(torch.nn.Module):
    """ Implementation of multinomial softmax regression using PyTorch """
    def __init__(self, input_dim, output_dim, lrate=1e-2, batch_size=256, epochs=1):
        super().__init__()
        self._linear = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 72),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(72, 72),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(72, output_dim)
        )
        self._lrate = lrate
        self._batch_size = batch_size
        self._epochs = epochs

    def forward(self, X):
        return self._linear(X)

    def fit(self, X, y, weights=None):
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
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

    def predict_proba(self, x, temp=1):
        """ Predicts probability of action classes under policy
        :param x:     Tensor of states of shape (n_states, state_dims)
        :param temp:  Temperature parameter to reduce overconfident predictions
        :returns:     Action probability matrix of shape (n_states, n_actions)
        """
        x = torch.tensor(x).float()
        logits = self(x)
        return torch.softmax(self(x) / temp, dim=1).detach().numpy()


def get_class_weights(df):
    """
    Weigh action classes inversely proportional to their frequency
    :param df:  DataFrame with training data
    :return:    Tensor of class weights
    """
    print('\nComputing action class weights')
    # Compute frequencies of actions relative to most freq class
    counts = df.action.value_counts()
    weights = torch.zeros(25)
    for i, count in counts.to_dict().items():
        weights[int(i)] = (1 / count) * (counts.sum() / 25)

    print('Action class weights:')
    print(weights, '\n')
    return weights.detach()


def estimate_behavior_policy(train_set, epochs=20, lrate=1e-2, batch_size=128):
    """ Convenience function for behavior policy estimation
    :param train_set:  Path to preprocessed training set with states and chosen actions (by physician)
    :returns:          Fitted Policy instance
    """
    # Load training set into DataFrame
    train_df = pd.read_csv(train_set)
    states = train_df.filter(regex='x\d+').values
    actions = train_df.action.values

    # class weights to handle action class imbalance
    weights = get_class_weights(train_df)

    print('\nFitting policy to %s states...' % states.shape[0])

    lr = Policy(input_dim=states.shape[1], output_dim=25, lrate=lrate, epochs=epochs, batch_size=batch_size)
    lr.fit(states, actions, weights=weights)
    return lr


def evaluate_policy(policy, dataset, temp=1, batch_size=128, add_true_action=True, verbose=True):
    """ Applies policy to held-out dataset and estimates distribution over actions
    :param policy:      kNN model fitted to physician actions
    :param dataset:     Dataset to evaluate policy on
    :param temp:        Temperature of softmax
    :param batch_size:  Size of batches used for evaluation (default: 128)
    """
    def onehot(x, num_classes):
        x = torch.tensor(x).long()
        onehot = torch.nn.functional.one_hot(x, num_classes=num_classes)
        return onehot.detach().numpy()

    # Load train/valid/test set into DataFrame
    df = pd.read_csv(dataset)
    episode = df.episode.values
    actions = df.action.values
    rewards = df.reward.values
    states = df.filter(regex='x\d+').values
    indices = df.index.values

    action_probs = []

    # Process batch-wise to reduce overload
    if verbose:
        print('\nProcessing %s\n' % dataset)

    for i in tqdm(range(0, len(indices), batch_size), disable=not verbose):
        index_batch = indices[i: i + batch_size]

        # Look up actions of k-nearest states in training set
        batch_action_probs = policy.predict_proba(states[index_batch], temp=temp)

        # Interpolate with chosen action by 10% to ensure a non-zero probability
        # is given to the action chosen by the physician
        if add_true_action:
            batch_action_onehot = onehot(actions[index_batch], num_classes=25)
            batch_action_probs = 0.9 * batch_action_probs + 0.1 * batch_action_onehot

        action_probs.append(batch_action_probs)

    # Stack batches
    action_probs = np.concatenate(action_probs, axis=0)

    # Combine action probs with chosen actions and rewards
    res = pd.DataFrame(data=np.column_stack([episode, actions, rewards, action_probs]),
                       index=indices, columns=['episode', 'action', 'reward'] + list(range(25)))
    return res


def reliability_score(action_probs, true_actions, bins=100):
    # estimate accuracy for each confidence bin
    # For details, see (Guo et al., 2017): https://arxiv.org/pdf/1706.04599.pdf
    predicted_actions = np.argmax(action_probs, axis=1)
    action_probs_binned = (action_probs.max(axis=1) * bins).round() / bins
    accuracies, confidences, support = [], [], []
    for conf in set(action_probs_binned):
        mask = action_probs_binned == conf
        acc = accuracy_score(true_actions[mask], predicted_actions[mask])
        accuracies.append(acc)
        confidences.append(conf)
        support.append(np.sum(mask))
    return confidences, accuracies, support


def calibrate_policy(policy, valid_file, n_actions, tau=1.0, iterations=50, bins=100, batch_size=128):
    # keep track of ece score
    start_ece = np.inf
    best_ece = np.inf

    # iteratively improve tau using hill-climbing
    print('Calibrating temperature...')
    with tqdm(total=iterations) as pbar:
        for i in range(iterations):
            # update tau with gaussian noise once baseline tau is scored
            tau2 = tau + (np.random.normal(0, 1) if i > 0 else 0)

            # action probs according to policy
            action_probs = evaluate_policy(policy, valid_file, temp=tau2, verbose=False)
            true_actions = action_probs.action.values
            action_probs = action_probs.filter(regex='\d+').values

            # expected calibration error (ECE)
            conf, acc, supp = reliability_score(action_probs, true_actions, bins=25)
            ece_score = sum([(s / sum(supp)) * abs(c - a) for c, a, s in zip(conf, acc, supp)])

            if ece_score < best_ece:
                if best_ece > 1e10:
                    start_ece = ece_score
                best_ece = ece_score
                tau = tau2

            pbar.update(1)
            pbar.set_postfix({'ECE_start': start_ece, 'ECE': best_ece, 'Tau': tau})

    return tau


def reliability_diagram(reliability_scores):
    # For details, see (Guo et al., 2017): https://arxiv.org/pdf/1706.04599.pdf
    confidences, accuracies, _ = reliability_scores
    plt.figure(figsize=(8, 6))
    plt.plot(confidences, accuracies, '.')
    plt.plot([0, 1], [0, 1], '--', c='gray')
    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')
    plt.show()


if __name__ == '__main__':
    DATASET = '../../preprocessing/datasets/amsterdam-umc-db_v2/aggregated_full_cohort_2h'

    #######################
    #   Estimate Policy   #
    #######################

    # Step 0. create output directory, e.g. mimic-iii-aggregated_1h
    path = PurePath(DATASET)
    out_dir = '%s_%s_mlp' % (path.parent.name, path.name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Step 1. estimate behavior policy from training set
    train_file = os.path.join(DATASET, 'train.csv')
    policy = estimate_behavior_policy(train_file)

    # Step 2. calibrate policy using per-action temperature scaling on validation set
    valid_file = os.path.join(DATASET, 'valid.csv')
    taus = calibrate_policy(policy, valid_file, n_actions=25)

    # Step 3. compute action distribution of behavior policy over each dataset
    for part in ['valid', 'test']:
        data_file = os.path.join(DATASET, part + '.csv')
        action_probs = evaluate_policy(policy, data_file, temp=taus)

        outfile = os.path.join(out_dir, '%s_behavior_policy.csv' % part)
        action_probs.to_csv(outfile, index=False)

    #####################
    #   Sanity Checks   #
    #####################

    # Is estimated policy predictive of the true policy's actions?
    valid_file = os.path.join(DATASET, 'valid.csv')
    valid_behavior_policy_file = os.path.join(out_dir, 'valid_behavior_policy.csv')

    true_actions = pd.read_csv(valid_file).action
    action_probs = pd.read_csv(valid_behavior_policy_file).filter(regex='\d+').values
    predicted_actions = np.argmax(action_probs, axis=1)

    print('Predictive of its own actions?')
    print('F1: ', f1_score(true_actions, predicted_actions, average='macro'))
    print('Acc:', accuracy_score(true_actions, predicted_actions))

    # Reliability diagrams
    reliability = reliability_score(action_probs, true_actions)
    reliability_diagram(reliability)






