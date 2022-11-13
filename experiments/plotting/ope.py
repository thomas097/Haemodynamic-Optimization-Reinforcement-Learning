import os
import pandas as pd
from importance_sampling import WeightedIS
from doubly_robust import WeightedDoublyRobust
from utils import *


def main(in_dir, model_paths, behavior_policy_file, mdp_training_file, method='fqe'):
    # Build WIS estimator
    wis = WeightedIS(behavior_policy_file, verbose=True)

    # Estimate WIS/WDR of physician policy (using training set to fit model)
    phys_action_probs = pd.read_csv(behavior_policy_file).filter(regex='\d+').values
    #wdr = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method=method).fit(phys_action_probs)
    phys_wdr_score = 0.0#wdr(phys_action_probs)
    phys_wis_score = wis(phys_action_probs)

    # Store results in DataFrame
    table = pd.DataFrame([[phys_wis_score], [phys_wdr_score]], columns=['Physician'], index=['WIS', 'WDR'])

    for model_name, (model_path, dataset_path) in model_paths.items():
        # Load model
        policy = load_pretrained(os.path.join(in_dir, model_path), 'policy.pt')
        encoder = load_pretrained(os.path.join(in_dir, model_path), 'encoder.pt')

        # Evaluate model on dataset
        dataset = pd.read_csv(dataset_path)
        action_probs = evaluate_policy_on_dataset(encoder, policy, dataset, _type='action_probs')

        # Build and train WDR estimator on evaluation policy
        #wdr = WeightedDoublyRobust(behavior_policy_file, mdp_training_file, method=method).fit(action_probs)

        # Compute WDR and WIS estimates of V^{pi_e}
        table[model_name] = [wis(action_probs), 0.0]#wdr(action_probs)]

    print(table)


if __name__ == '__main__':
    roggeveen_data_file = '../../preprocessing/datasets/mimic-iii/roggeveen_4h/mimic-iii_valid.csv'
    attention_data_file = '../../preprocessing/datasets/mimic-iii/attention_4h/mimic-iii_valid.csv'
    behavior_policy_file = '../../ope/physician_policy/roggeveen_4h/mimic-iii_valid_behavior_policy.csv'

    paths = {'Roggeveen et al.': ('roggeveen_experiment_00000', roggeveen_data_file),
             }

    in_dir = '../results/'

    main(in_dir, paths, behavior_policy_file, roggeveen_data_file)
