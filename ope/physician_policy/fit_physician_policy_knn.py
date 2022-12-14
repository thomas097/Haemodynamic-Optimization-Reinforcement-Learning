import os
import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path, PurePath


class FaissWeightedKNN:
    """ Fast implementation of k-Nearest Neighbors using faiss """
    def __init__(self, weights, actions, k=5, nprobe=10, decay_rate=1.0):
        self._index = None
        self._actions = actions
        self._weights = weights[np.newaxis]
        self._nprobe = nprobe
        self._decay_rate = decay_rate   # to emphasize states that are closest
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
        self._index.nprobe = self._nprobe
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

        # Convert distance to similarity
        weights = 1 / (self._decay_rate * distances + 1)

        # Compute for each action the total sum of actions weighted by the distance of their respective states
        action_count = np.zeros((X.shape[0], len(self._actions)))
        for i, a in enumerate(self._actions):
            action_count[:, i] = np.sum(weights * (actions == a), axis=1)

            # add current action too to ensure non-zero prob for chosen action
            if true_action is not None:
                action_count[:, i] += (true_action == a)

        # Normalize to obtain probabilities
        return action_count / np.sum(action_count, axis=1, keepdims=True)


def get_feature_weights(features_file, special_features, special_weight=2):
    # associate feature names with index in feature vector
    with open(features_file, 'r', encoding='utf-8') as file:
        feature_indices = {line.strip(): i for i, line in enumerate(file.readlines())}

    # Create vector of feature weights taking into account special weights
    weights = np.ones(len(feature_indices))
    for f in special_features:
        weights[feature_indices[f]] = special_weight
    return weights


def estimate_behavior_policy(train_set, weights, k=150):
    """ kNN-based behavior policy estimation for OPE. Please refer to (Raghu et al., 2018) for details.

        train_set:     Path to preprocessed training set with states and chosen actions (by physician)
        weights:       Importance weights to assign to each state-space feature

        returns:       Fitted kNN model
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
    print('\nProcessing %s\n' % dataset)
    for i in tqdm(range(0, len(indices), batch_size)):
        index_batch = indices[i: i + batch_size]

        # Look up actions of k-nearest states in training set
        batch_action_probs = policy.predict_proba(states[index_batch], true_action=actions[index_batch])
        action_probs.append(batch_action_probs)

    # Stack batches
    action_probs = np.concatenate(action_probs, axis=0)

    # Combine action probs with chosen actions and rewards
    res = pd.DataFrame(data=np.column_stack([episode, actions, rewards, action_probs]),
                       index=indices, columns=['episode', 'action', 'reward'] + list(range(25)))
    return res


if __name__ == '__main__':
    DATASET = '../../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h'

    # Assign certain features additional weight. Please refer to (Roggeveen et al., 2021)
    # Remark: In Roggeveen et al. and Raghu et al., `chloride` is said to be up-weighted, but not in the code (Why?)
    SPECIAL_FEATURES = ['total_iv_fluid_prev', 'sofa_score', 'weight', 'age', 'mean_bp',
                        'dias_bp', 'chloride', 'lactate', 'pf_ratio', 'total_urine_output']
    weights = get_feature_weights(os.path.join(DATASET, 'state_space_features.txt'), SPECIAL_FEATURES)

    #######################
    #   Estimate Policy   #
    #######################

    # Step 0. create output directory, e.g. mimic-iii-aggregated_1h
    path = PurePath(DATASET)
    out_dir = '%s_%s_knn' % (path.parent.name, path.name)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # Step 1. estimate behavior policy from training set
    train_file = os.path.join(DATASET, 'train.csv')
    policy = estimate_behavior_policy(train_file, weights=weights)

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






