import os
import pickle
import torch
import numpy as np

from experience_replay import EvaluationReplay


def load_actions_to_bins(path):
    with open(path, 'rb') as file:
        return pickle.load(file)


def load_pretrained(path, model):
    """ Loads pretrained model from file """
    if model not in ['encoder.pkl', 'policy.pkl']:
        raise Exception("Invalid model argument; Choose 'encoder.pkl' or 'policy.pkl'")

    full_path = os.path.join(path, model)
    if not os.path.exists(full_path):
        if model == 'policy.pkl':
            raise Exception('No policy.pkl in %s ' % path)
        return None

    with open(full_path, 'rb') as file:
        model = pickle.load(file)
    return model


def evaluate_on_dataset(encoder, policy, dataset, _type='qvals'):
    # Load dataset into replay buffer
    replay = EvaluationReplay(dataset, return_history=encoder is not None)

    # Feed histories through (identity) encoder to get fixed state representation
    encoder = encoder if encoder is not None else (lambda x: x)
    encoded_states = torch.concat([encoder(t) for t in replay.iterate()])

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
    return torch.concat([encoder(t) for t in replay.iterate()]).detach().numpy()