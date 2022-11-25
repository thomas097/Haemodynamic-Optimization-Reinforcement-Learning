import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from per_horizon_importance_sampling import PHWIS
from doubly_robust import WeightedDoublyRobust
from experience_replay import EvaluationReplay


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


def get_action_probs(policy, dataset_file, batch_size):
    """ Obtains action probabilities for each history in dataset
    :param policy:      A trained policy network
    :param dataset:     Dataset with variable-length patient trajectories
    :param batch_size:  Number of histories to process at once
    :returns:           Tensor of shape (n_states, n_actions)
    """
    # load dataset into replay buffer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay = EvaluationReplay(pd.read_csv(dataset_file), device=device)

    with torch.no_grad():
        action_probs = []
        with tqdm(total=len(dataset), desc='gathering predictions...', position=0, leave=True) as pbar:
            for histories in replay.iterate(batch_size):
                # histories -> encoder -> policy network + softmax
                probs = torch.softmax(model(histories), dim=1).cpu().detach().numpy()
                action_probs.append(probs)
                pbar.update(histories.size(0))

    # convert torch tensors to numpy ndarray
    return np.concatenate(action_probs, axis=0)


def evaluate_policy(policy_file, dataset_file, behavior_policy_file, batch_size=256, lrate=1e-2, iters=1000):
    """ Computes WIS, FQE and WDR estimates of policy performance
    :param policy_file:           A trained policy network
    :param dataset_file:          Path to dataset with variable-length patient trajectories
    :param behavior_policy_file:  Path to estimated behavior policy file containing action
                                  probabilities at each state in the dataset
    :param batch_size:            Number of histories to process at once
    :param n_bootstraps:
    """
    # get policy's action probs for states in dataset
    action_probs = get_action_probs(policy=load_pretrained(policy_file), dataset_file=dataset_file, batch_size=batch_size)

    # fit WDR's DM estimator to policy
    wdr = WeightedDoublyRobust(behavior_policy_file, mdp_training_file=dataset_file, lrate=lrate, iters=iters)
    wdr.fit(action_probs)







if __name__ == '__main__':
    evaluate_policy(
        policy_file='../results/transformer_experiment_00000/model_5000.pt',
        dataset_file='../../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_1h/valid.csv',
        behavior_policy_file='../../ope/physician_policy/amsterdam-umc-db_aggregated_full_cohort_1h_knn/valid_behavior_policy.csv',
    )
