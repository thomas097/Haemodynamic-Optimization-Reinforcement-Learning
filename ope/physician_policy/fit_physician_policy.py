import os
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path


class FaissWeightedKNN:
    """ Fast implementation of k-Nearest Neighbors using faiss """
    def __init__(self, weights, actions, k=5):
        self._index = None
        self._actions = actions
        self._weights = weights[np.newaxis]
        self._y = None
        self._k = k

    def _apply_weights(self, X):
        """ As Faiss does not support weighted kNN by default, we can use
            the identity w(x-y)^2 = (sqrt(w)(x-y))^2 = (sqrt(w)x - sqrt(w)y)^2 and
            add the weights as sqrt(weights) to X directly."""
        return (X * np.sqrt(self._weights)).astype(np.float32)

    def fit(self, X, y):
        self._y = y
        self._index = faiss.IndexFlatL2(X.shape[1])
        self._index.add(self._apply_weights(X))

    def predict(self, X):
        # Determine indices of similar states in X_train
        _, indices = self._index.search(self._apply_weights(X), k=self._k)
        return self._y[indices]

    def predict_proba(self, X, true_action=None):
        # Determine indices and distances of similar states in X_train
        distances, indices = self._index.search(self._apply_weights(X), k=self._k)

        # Determine actions associated with states
        actions = self._y[indices]

        # Convert distances to similarity
        weights = np.exp(-distances)

        # Compute for each action the total sum of actions weighted by the distance of their respective states
        action_count = np.zeros((X.shape[0], len(self._actions)))
        for i, a in enumerate(self._actions):
            action_count[:, i] = np.sum(weights * (actions == a), axis=1)

            # Add true action to it with a weight of 1 if given
            if true_action is not None:
                action_count[:, i] += (true_action == a).flatten()

        # Normalize to obtain probabilities
        return action_count / np.sum(action_count, axis=1, keepdims=True)


def get_feature_weights(feature_names, feature_weights):
    # Create vector of feature weights taking into account special feature weights
    weights = np.ones(len(feature_names))
    for feat, weight in feature_weights.items():
        if feat not in feature_names:
            raise Exception('Feature %s not in feature_names' % feature_names)
        weights[feature_names.index(feat)] = weight
    return weights


def estimate_behavior_policy(train_set, weights, k=300):
    """ kNN-based behavior policy estimation for OPE. Please refer to (Raghu et al., 2018) for details.

        train_set:     Path to preprocessed training set with states and chosen actions (by physisian)
        weights:       Importance weights to assign to each state-space feature

        returns:       Fitted kNN model and list of chosen actions (i.e. train_set[action_col])
    """
    # Load training set into DataFrame
    train_df = pd.read_csv(train_set)
    states = train_df.filter(regex='x\d+').values
    actions = train_df['action'].values

    i = np.arange(len(states))
    print('\nFitting KNN policy to %s states...' % len(i))

    knn = FaissWeightedKNN(weights=weights, actions=np.arange(25), k=k)
    knn.fit(states[i], actions[i])
    print('Done!')
    return knn


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
    print('Processing %s\n' % dataset)
    for i in tqdm(range(0, len(indices), batch_size)):
        index_batch = indices[i: i + batch_size]

        # Look up actions of k-nearest states in training set
        batch_action_probs = policy.predict_proba(states[index_batch], true_action=actions[index_batch])
        action_probs.append(batch_action_probs)

    # Stack batches
    action_probs = np.concatenate(action_probs, axis=0)

    # Combine action probs with chosen actions and rewards
    return pd.DataFrame(data=np.column_stack([episode, actions, rewards, action_probs]),
                        index=indices, columns=['episode', 'action', 'reward'] + list(range(25)))


if __name__ == '__main__':
    # Estimate behavior policy from training set
    TRAIN_SET = '../../preprocessing/datasets/mimic-iii/roggeveen_4h_with_cv/mimic-iii_train.csv'
    OUT_DIR = 'roggeveen_4h_with_cv'

    # Evaluate policy's actions on train/valid/test sets
    DATASETS = ['../../preprocessing/datasets/mimic-iii/roggeveen_4h_with_cv/mimic-iii_train.csv',
                '../../preprocessing/datasets/mimic-iii/roggeveen_4h_with_cv/mimic-iii_valid.csv',
                '../../preprocessing/datasets/mimic-iii/roggeveen_4h_with_cv/mimic-iii_test.csv']

    # Assign certain features additional weight. Please refer to (Roggeveen et al., 2021).
    # Remark: In the original work, `chloride` is also up-weighted, but not in the code (why?)
    SPECIAL_WEIGHTS = {'x7': 2,   # age
                       'x29': 2,  # lactate
                       'x43': 2,  # pf_ratio
                       'x3': 2,   # sofa_score
                       'x4': 2,   # weight
                       'x46': 2,  # total_urine_output
                       'x12': 2,  # dias_bp
                       'x11': 2,  # mean_bp
                       'x1': 2}   # total_iv_fluid
    STATE_COLS = ['x%d' % i for i in range(48)]

    #######################
    #   Estimate Policy   #
    #######################

    # Step 0. Create output directory
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Step 1. Estimate behavior policy from training set
    weights = get_feature_weights(STATE_COLS, SPECIAL_WEIGHTS)
    policy = estimate_behavior_policy(TRAIN_SET, weights=weights)

    # Step 2. Estimate action distribution of behavior policy over each dataset
    for dataset in DATASETS:
        action_probs = evaluate_policy(policy, dataset)

        outfile = os.path.join(OUT_DIR, Path(dataset).stem)
        action_probs.to_csv(outfile + '_behavior_policy.csv', index=False)




