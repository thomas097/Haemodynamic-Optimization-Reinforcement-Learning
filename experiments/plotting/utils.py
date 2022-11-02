import os
import io
import pickle
import torch
import numpy as np

from experience_replay import EvaluationReplay


def load_actions_to_bins(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def load_pretrained(path, model):
    """ Loads pretrained model from file """
    if model not in ['encoder.pt', 'policy.pt']:
        raise Exception("Invalid model argument; Choose 'encoder.pt' or 'policy.pt'")

    full_path = os.path.join(path, model)
    if not os.path.exists(full_path):
        if model == 'policy.pt':
            raise Exception('No policy.pt in %s ' % path)
        return None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return torch.load(full_path, map_location=device)


def evaluate_policy_on_dataset(encoder, policy, dataset, _type='qvals'):
    # Load dataset into replay buffer
    replay = EvaluationReplay(dataset, return_history=encoder is not None)

    # Feed histories through encoder to get fixed state representation
    if encoder is not None:
        encoded_states = torch.concat([encoder(t).detach() for t in replay.iterate()])
    else:
        encoded_states = torch.concat([t for t in replay.iterate()])

    # Return Q-values according to model
    if _type == 'qvals':
        return policy(encoded_states).detach().numpy()
    elif _type == 'actions':
        return policy.sample(encoded_states)
    elif _type == 'action_probs':
        return policy.action_probs(encoded_states)
    else:
        raise Exception("_type %s not understood" % _type)


def run_encoder_over_dataset(encoder, dataset):
    replay = EvaluationReplay(dataset, return_history=True)
    return torch.concat([encoder(t).detach() for t in replay.iterate()]).detach().numpy()
