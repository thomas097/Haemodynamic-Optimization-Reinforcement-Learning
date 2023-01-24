import os
import random
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from per_horizon_importance_sampling import PHWIS
from doubly_robust import WeightedDoublyRobust
from experience_replay import EvaluationReplay


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


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


def get_action_probs(policy, dataset_file, batch_size=256):
    """ Obtains action probabilities for each history in dataset
    :param policy:      A trained policy network
    :param dataset:     Dataset with variable-length patient trajectories
    :param batch_size:  Number of histories to process at once
    :returns:           Tensor of shape (n_states, n_actions)
    """
    # load dataset into replay buffer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset = pd.read_csv(dataset_file)
    replay = EvaluationReplay(dataset=dataset, device=device)

    with torch.no_grad():
        action_probs = []
        with tqdm(total=len(dataset), desc='gathering predictions', position=0, leave=True) as pbar:
            for histories in replay.iterate(batch_size):
                # histories -> encoder -> policy network + softmax
                qvals = policy(histories)
                probs = torch.softmax(qvals, dim=1).cpu().detach().numpy()
                action_probs.append(probs)
                pbar.update(histories.size(0))

    # convert torch tensors to numpy ndarray
    return np.concatenate(action_probs, axis=0)


def conf_interval(scores, conf_level):
    """ Determines the upper and lower confidence bound on the median score
    using bootstrapped performance samples in `scores`
    :param scores:      Performance scores obtain through bootstrapping
    :param conf_level:  Confidence level (default: 0.95)
    """
    median = np.median(scores)
    lower = np.quantile(scores, q=(1 - conf_level) / 2)
    upper = np.quantile(scores, q=conf_level + (1 - conf_level) / 2)
    return '%.3f (±%.3f) [%.3f - %.3f]' % (median, upper - median, lower, upper)


def evaluate_policy(policy_file, dataset_file, behavior_policy_file, batch_size=256, lrate=1e-2, iters=15000,
                    gamma=1.0, n_bootstraps=100, fraction=0.8, conf_level=0.95, seed=42):
    """ Computes WIS, FQE and WDR estimates of policy performance
    :param policy_file:           File pointing to a trained policy network. If policy_file == behavior_policy_file,
                                  then the estimated behavior policy is evaluated.
    :param dataset_file:          Path to dataset with variable-length patient trajectories
    :param behavior_policy_file:  Path to estimated behavior policy file containing action
                                  probabilities at each state in the dataset
    :param batch_size:            Number of histories to process at once (default: 256)
    :parma lrate:                 Learning rate used to estimate WDR's FQE estimator
    :param iters:                 Number of training iterations used to estimate WDR's FQE estimator
    :param n_bootstraps:          Number of bootstrap sets to sample to estimate confidence bounds (default: 100)
    :param fraction:              Fraction of episodes to construct each bootstrap set (default: 0.8)
    :param conf_level:            Confidence level (default: 0.95)
    """
    print('Evaluating:', policy_file)
    # make evaluation reproducible
    random.seed(seed)

    # get policy's action probs for states in dataset
    if policy_file != behavior_policy_file:
        action_probs = get_action_probs(policy=load_pretrained(policy_file), dataset_file=dataset_file, batch_size=batch_size)
    else:
        print('Evaluating physician policy...')
        action_probs = pd.read_csv(behavior_policy_file).filter(regex='\d+').values

    # fit WDR's FQE estimator to policy
    wdr = WeightedDoublyRobust(behavior_policy_file, mdp_training_file=dataset_file, method='fqe', lrate=lrate,
                               gamma=gamma, iters=iters)
    print('PHWIS (ESS):', wdr.phwis(action_probs), '(%.2f)' % wdr.phwis.ess)
    wdr.fit(action_probs)

    # list episodes from which to bootstrap
    dataset = pd.read_csv(dataset_file)
    episodes = dataset.episode.unique().tolist()
    print('Total number of episodes:', len(episodes))

    phwis_scores = []
    wdr_scores = []
    fqe_scores = []
    sample_sizes = []

    for _ in tqdm(range(n_bootstraps), desc='Bootstrapping'):
        # sample bootstrap set
        bootstrap_set = random.sample(episodes, k=int(fraction * len(episodes)))

        # metrics
        phwis_score = wdr.phwis(action_probs, episodes=bootstrap_set)
        ess_score = wdr.phwis.ess
        wdr_score, fqe_score = wdr(action_probs, episodes=bootstrap_set) # FQE is computed as part of WDR computation

        # Limit fqe to starting states only
        bootstrap_dataset = dataset[dataset.episode.isin(bootstrap_set)].copy().reset_index(drop=True)
        first_state_idx = [ep.index[0] for _, ep in bootstrap_dataset.groupby('episode', sort=False)]
        fqe_score = np.mean(fqe_score[first_state_idx])

        phwis_scores.append(phwis_score)
        wdr_scores.append(wdr_score)
        fqe_scores.append(fqe_score)
        sample_sizes.append(ess_score)

    # show results in stdout
    result = pd.DataFrame({
        'PHWIS': [conf_interval(phwis_scores, conf_level=conf_level)],
        'PHWDR': [conf_interval(wdr_scores, conf_level=conf_level)],
        'FQE': [conf_interval(fqe_scores, conf_level=conf_level)],
        'ESS': [np.median(sample_sizes)]
    }, index=[''])
    print(result)


def save_tail(path, fname, maxlen=256):
    df = pd.read_csv(path)
    df = df.groupby('episode', as_index=False, sort=False).tail(maxlen)
    df.to_csv(fname, index=False)
    return fname


if __name__ == '__main__':
    dataset_file = '../../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_2h/valid.csv'
    behavior_policy_file = '../../ope/physician_policy/amsterdam-umc-db_aggregated_full_cohort_2h_mlp/valid_behavior_policy.csv'
    model = '../results/amsterdam-umc-db/transformer_experiment_00000/model.pt'

    # Truncate episodes to fix degeneracy problem
    save_tail(dataset_file, 'valid_tmp.csv', maxlen=32)
    save_tail(behavior_policy_file, 'behavior_policy_tmp.csv', maxlen=32)

    evaluate_policy(
        policy_file=model,
        dataset_file='valid_tmp.csv',
        behavior_policy_file='behavior_policy_tmp.csv',
        lrate=0.05,
        iters=1000,
        gamma=0.95,
        n_bootstraps=1000,
        fraction=0.75,
    )
