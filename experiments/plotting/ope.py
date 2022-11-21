import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from importance_sampling import WIS
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


def evaluate(model, dataset, behavior_policy_file, mdp_training_file, batch_size=256):
    # load dataset into replay buffer
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    replay = EvaluationReplay(dataset, device=device)

    with torch.no_grad():
        action_probs = []
        with tqdm(total=len(dataset), desc='gathering predictions', position=0, leave=True) as pbar:
            for histories in replay.iterate(batch_size):
                # histories -> encoder -> policy network + softmax
                probs = torch.softmax(model(histories), dim=1).cpu().detach().numpy()
                action_probs.append(probs)
                pbar.update(histories.size(0))

    # convert torch tensors to numpy ndarray
    policy = np.concatenate(action_probs, axis=0)

    # weighted importance sampling
    wis = WIS(behavior_policy_file, bootstraps=1000)
    wis_low, wis_mid, wis_high = wis(policy)
    print('WIS = %.3f [%.3f, %.3f]' % (wis_mid, wis_low, wis_high))

    # doubly robust
    wdr = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method='fqe').fit(policy)
    print('WDR:  ', wdr(policy))


if __name__ == '__main__':
    model = load_pretrained('../results/transformer_experiment_00000/model_5000.pt')
    dataset = pd.read_csv('../../preprocessing/datasets/mimic-iii/aggregated_all_1h/mimic-iii_valid.csv')

    evaluate(
        model=model,
        dataset=dataset,
        behavior_policy_file='../../ope/physician_policy/aggregated_all_1h/mimic-iii_valid_behavior_policy.csv',
        mdp_training_file='../../preprocessing/datasets/mimic-iii/aggregated_all_1h/mimic-iii_valid.csv'
    )
