import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from sklearn.neighbors import KNeighborsClassifier


class WeightedMinkowski:
    """ Implementation of the distance metric used by (Roggeveen et al., 2021)
        for the KNN-based estimation of the behavior (physician) policy for OPE.
    """
    def __init__(self, feature_names, feature_weights):
        # Create vector of feature weights taking into account special feature weights
        self._weights = np.ones(len(feature_names))
        for feat, weight in feature_weights.items():
            if feat not in feature_names:
                raise Exception('Feature %s not in feature_names' % feature_names)
            self._weights[feature_names.index(feat)] = weight

    def __call__(self, x, y):
        return np.linalg.norm((x - y) * self._weights)


def estimate_behavior_policy(train_set, state_cols, action_col, metric, num_neighbors=300):
    """ kNN-based behavior policy estimation for OPE. Please refer to (Raghu et al., 2018) for details.
    """
    # Load training set into DataFrame
    train_df = pd.read_csv(train_set, usecols=state_cols + [action_col])
    states = train_df[state_cols].values
    actions = train_df[action_col].values

    print('\nFitting KNN policy...')
    knn = KNeighborsClassifier(n_neighbors=num_neighbors, metric=metric, algorithm='auto')
    knn.fit(states, actions)
    print('Done!')
    return knn, actions


def evaluate_policy(policy, train_actions, dataset, state_cols, action_col, reward_col, batch_size=128):
    """ Applies policy to held-out dataset and estimates distribution over actions.
    """
    # Load train/valid/test set into DataFrame
    df = pd.read_csv(dataset, usecols=state_cols + [action_col, reward_col])
    indices = df.index.values
    actions = df[action_col].values
    rewards = df[reward_col].values
    states = df[state_cols].values

    all_action_probs = []

    # Process batch-wise to reduce overload
    print('Processing %s\n' % dataset)
    for i in tqdm(range(0, len(indices), batch_size)):
        index_batch = indices[i: i + batch_size]

        # Look up actions of k-nearest states in training set
        neigh_indices = policy.kneighbors(states[index_batch], return_distance=False)
        neigh_actions = train_actions[neigh_indices]

        # Add action actually chosen by policy
        neigh_actions = np.column_stack([neigh_actions, actions[index_batch]])

        # Convert actions incidence (0-24) to probabilities
        k = neigh_actions.shape[1]
        action_probs = np.column_stack([np.sum(neigh_actions == a, axis=1) / k for a in range(25)])
        all_action_probs.append(action_probs)

    # Combine batches into DataFrame with action probabilities
    action_probs = pd.DataFrame(data=np.concatenate(all_action_probs, axis=0),
                                columns=range(25), index=indices)

    return action_probs, actions, rewards


if __name__ == '__main__':
    # Estimate behavior policy from training set
    TRAIN_SET = '../../preprocessing/datasets/mimic-iii/roggeveen/mimic-iii_train.csv'

    # Evaluate policy's actions on train/valid/test sets
    DATASETS = ['../../preprocessing/datasets/mimic-iii/roggeveen/mimic-iii_train.csv',
                '../../preprocessing/datasets/mimic-iii/roggeveen/mimic-iii_valid.csv',
                '../../preprocessing/datasets/mimic-iii/roggeveen/mimic-iii_test.csv']

    # Defines state and action space
    STATE_COLS = ['max_vp', 'total_iv_fluid', 'sirs_score', 'sofa_score', 'weight', 'ventilator', 'height', 'age',
                  'gender', 'heart_rate', 'temp', 'mean_bp', 'dias_bp', 'sys_bp', 'resp_rate', 'spo2', 'natrium',
                  'chloride', 'kalium', 'trombo', 'leu', 'anion_gap', 'aptt', 'art_ph', 'asat', 'alat', 'bicarbonaat',
                  'art_be', 'ion_ca', 'lactate', 'paco2', 'pao2', 'shock_index', 'hb', 'bilirubin', 'creatinine',
                  'inr', 'ureum', 'albumin', 'magnesium', 'calcium', 'pf_ratio', 'glucose', 'total_urine_output',
                  'running_total_urine_output', 'running_total_iv_fluid']
    ACTION_COL = 'action'
    REWARD_COL = 'reward'

    # Assign certain features additional weight. Please refer to (Roggeveen et al., 2021).
    SPECIAL_WEIGHTS = {'age': 2, 'chloride': 2, 'lactate': 2, 'pf_ratio': 2, 'sofa_score': 2, 'weight': 2,
                       'total_urine_output': 2, 'dias_bp': 2, 'mean_bp': 2}

    #######################
    #   Estimate Policy   #
    #######################

    OUT_DIR = 'physician_policy/'
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    # Step 1. Estimate behavior policy from training set
    weighted_dist = WeightedMinkowski(STATE_COLS, SPECIAL_WEIGHTS)
    policy, train_actions = estimate_behavior_policy(TRAIN_SET, STATE_COLS, ACTION_COL, metric=weighted_dist)

    # Step 2. Estimate action distribution of behavior policy over each dataset
    for dataset in DATASETS:
        action_probs, actions, rewards = evaluate_policy(policy, train_actions, dataset, STATE_COLS, ACTION_COL, REWARD_COL)

        outfile = OUT_DIR + Path(dataset).stem
        np.savetxt(outfile + '_action_probs.csv', action_probs)
        np.savetxt(outfile + '_actions.csv', actions)
        np.savetxt(outfile + '_rewards.csv', rewards)




