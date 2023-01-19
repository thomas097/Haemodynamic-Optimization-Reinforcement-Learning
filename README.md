# Reinforcement learning with self-supervised state representations for hemodynamic support at the ICU $^\star$

This repository contains the source code accompanying the MSc thesis <i>"Reinforcement learning with self-supervised state representations for hemodynamic support at the ICU"</i> in partial fulfillment of the degree of MSc Artificial Intelligence at the Vrije Universiteit (VU), Amsterdam.

### Abstract
<i> In critically ill patients with sepsis or shock at the Intensive Care Unit (ICU), hemodynamic support through the administration of fluids and vasopressors can be essential to ensure adequate oxygen delivery to vital organs and prevent organ failure; 
however, identifying appropriate dosing parameters has proven challenging and, to date, no universally accepted protocol exists.
Recently, attempts have been made to learn treatment strategies for hemodynamic support with Reinforcement Learning (RL), with a particular focus on optimizing dosing strategies for the delivery of fluids and vasopressors from historical patient data. 
These methods rely on compact representations of the patient’s state to model all information on the patient needed to inform treatment decisions; 
however, designing such representations has remained an open problem. 
Inspired by recent work on state representation learning, this thesis explores the potential of artificial neural networks and self-supervised learning to learn representations of patient states for the purpose of optimizing treatment policies with reinforcement learning. 
To derive representations of patient states from observations and past treatments of a patient, two encoding architectures were developed based on a Continuous Kernel Convolutional Network (CKCNN) and a Causal Transformer and trained on a range of self-supervised and supervised objectives, including autoencoding, dynam-
ics modeling and behavior cloning, incentivizing the encoders to learn diverse, task-informative representations; 
these representations of patient states were then used for optimizing treatment policies for hemodynamic support with reinforcement learning. 
Results obtained using Off-Policy Policy Evaluation (OPE) on patient trajectories from the MIMIC-III and AmsterdamUMCdb intensive care databases demonstrate that policies learnt using self-supervised state representations have the potential to significantly improve treatment outcomes over policies learnt using conventional state spaces; 
however, despite the promising results, marked differences in strategies were observed between treatment policies, indicating a need for caution when interpreting treatment decisions</i>

## Dependencies
A Python installation 3.8+ with the following libraries installed:
- numpy
- matplotlib
- pandas
- torch
- tqdm
- seaborn

## Structure

The repository is organized as follows:

- `exploratory_analysis`: Exploratory Data Analysis (EDA) of the MIMIC-III and AmsterdamUMCdb intensive care databases

- `/data`: this folder contains all *data extraction* source code used to extract raw unprocessed patient data from MIMIC-III and the AmsterdamUMCdb part of the input (i.e. observation- and action-space), including vitals, lab results, demographics, treatment parameters, discharge information, etc.

- `preprocessing`: Code used to convert the raw measrurements and treatment parameters extracted from MIMIC-III and the AmsterdamUMCdb into regularly-sampled patient trajectories with 2h time steps.

- `model`: Implementations of the CKCNN and Transformer encoders and a range of baseline state space encoders (e.g. last observation, last K observatons, LSTM, etc.). For details, see the report. 

- `pretraining`: Implementation and training routines of the autoencoding, forward dynamics modeling, behavior cloning and combined 'multi-task' pretraining tasks 

- `experiments`: Training routines to learn treatment policies on MIMIC-III or the AmsterdamUMCdb with a (pre)trained encoder. Also, contains evaluation routines, e.g. to evaluate with OPE (see instructions below).

- `ope`: Definitions of the OPE estimators used to evaluate the learnt policies

## Instructions

### Reproducing the Reported Results
To reproduce the performance results with OPE of the report:

#### Preliminaries
1. Download models from **ADD LINK TO DRIVE!** and unpack in `experiments/results`
  -- You should have a folder structure as `experiments/results/<DATASET_NAME>/<EXPERIMENT_NAME>`
2. Download datasets from **ADD LINK TO DRIVE!** and unpack in `preprocessing/datasets`
3. Download pre-estimated behavior/physician policies from **ADD LINK TO DRIVE!** and unpack in `ope/physician_policy`  

#### Reproducing Evaluation Results
  
1. Navigate to `experiments/plotting/ope.py`:
  - Set `model` path to the location of the model to be evaluated, e.g. `../results/amsterdam-umc-db/transformer_experiment_00000/model.pt`
  - Set `dataset_file ` path to the location of the dataset to evaluate model on, e.g. '../../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_2h/test.csv'
  - Set `behavior_policy_file ` path to location of pre-estimated behavior policy, e.g. '../../ope/physician_policy/amsterdam-umc-db_aggregated_full_cohort_2h_mlp/test_behavior_policy.csv'
  
2. Run `ope.py`
  - This might take a while as an FQE model is fit onto the action distribution of the policy network
  
For physician's policy OPE results, replace `model` by filename of behavior policy file, i.e. `model = behavior_policy_file`

#### Action matrices

1. Navigate to `experiments/plotting/action_matrices.py`
  - Set `dataset_file ` path to the location of the dataset to evaluate model on, e.g. '../../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_2h/test.csv'
  - Set `dataset_label` to `mimic-iii` or `amsterdam-umc-db` (depending on the dataset chosen)
  
2. Run `action_matrices.py`
  
