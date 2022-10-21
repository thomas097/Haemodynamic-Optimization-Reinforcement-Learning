import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch.jit

from experience_replay import EvaluationReplay

sns.set_theme(style="darkgrid")


def get_physician_actions(data_path):
    """ Return table of shape (num_timesteps, num_admissions) with
        actions taken by physician ('0'-'24')
    """
    df = pd.read_csv(data_path)
    num_timesteps = int(df.groupby('icustay_id').size().max())
    return df['action'].values.reshape(-1, num_timesteps)


def get_model_actions(model_path, data_path, state_cols, episode_col):
    """ Feeds states through policy stored in `model_path`
        (and optionally an encoder if one was trained)
    """
    # Load policy
    policy_path = os.path.join(model_path, 'policy.pkl')
    with open(policy_path, 'rb') as file:
        policy = pickle.load(file)

    # Load encoder (if one was trained)
    encoder_path = os.path.join(model_path, 'encoder.pkl')
    encoder = lambda s: s  # Default to identity
    if os.path.exists(encoder_path):
        with open(encoder_path, 'rb') as file:
            encoder = pickle.load(file)

    # Create dataset iterator
    test_df = pd.read_csv(data_path)
    replay = EvaluationReplay(states=test_df[state_cols], episodes=test_df[episode_col],
                              return_history=encoder is not None)

    # Feed histories through encoder (or identity function)
    states = torch.concat([encoder(t) for t in replay.iterate()], dim=0)

    # Greed action selection
    actions = policy.sample(states)

    # Reshape to table of shape (num_admissions, num_timesteps)
    num_timesteps = int(test_df.groupby('icustay_id').size().max())
    return actions.reshape(-1, num_timesteps)


def plot_percentage_treatment(phys_actions, model_actions):
    # Drop absorbing terminal state
    phys_actions = phys_actions[:, :-1]
    model_actions = model_actions[:, :-1]

    # Compute expected non-zero actions at each timestep over admissions
    phys_perc_action = np.mean(phys_actions > 0, axis=0)
    model_perc_action = np.mean(model_actions > 0, axis=0)

    timesteps = np.arange(phys_actions.shape[1])
    plt.plot(timesteps, phys_perc_action, label='Physician')
    plt.plot(timesteps, model_perc_action, label='Model')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    # Define columns marking state- and action-space
    STATE_COLUMNS = ['max_vaso', 'total_iv_fluid', 'sirs_score', 'sofa_score', 'weight', 'ventilator', 'height', 'age',
                     'gender', 'heart_rate', 'temp', 'mean_bp', 'dias_bp', 'sys_bp', 'resp_rate', 'spo2', 'natrium',
                     'chloride', 'kalium', 'trombo', 'leu', 'anion_gap', 'aptt', 'art_ph', 'asat', 'fio2', 'alat',
                     'bicarbonaat', 'art_be', 'ion_ca', 'lactate', 'paco2', 'pao2', 'shock_index', 'hb', 'bilirubin',
                     'creatinine', 'inr', 'ureum', 'albumin', 'magnesium', 'calcium', 'pf_ratio', 'glucose',
                     'running_total_urine_output', 'total_urine_output', 'running_total_iv_fluid']

    # Physician (validation)
    phys_actions = get_physician_actions(data_path='../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv')

    # Model
    model_actions = get_model_actions(model_path='results/roggeveen_experiment_2022-10-17_14-16-05',
                                      data_path='../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv',
                                      state_cols=STATE_COLUMNS,
                                      episode_col='icustay_id')

    plot_percentage_treatment(phys_actions, model_actions)