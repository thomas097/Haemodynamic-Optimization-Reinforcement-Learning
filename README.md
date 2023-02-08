# Reinforcement Learning with Self-Supervised State Representations for Hemodynamic Support at the ICU

This repository contains the source code accompanying the MSc thesis <i>"Reinforcement learning with self-supervised state representations for hemodynamic support at the ICU"</i> in partial fulfillment of the degree of MSc Artificial Intelligence at the Vrije Universiteit (VU), Amsterdam.

<b>Abstract.</b> In critically-ill patients with sepsis or shock at the Intensive Care Unit (ICU), <b>hemodynamic support</b> through administration of fluids and vasopressors can be essential to maintain adequate circulation and oxygen delivery to vital organs and prevent organ failure; however, identifying appropriate treatment doses of fluids and vasopressors has proven challenging and, to date, no consensus on practice exists. Recently, efforts have been made to learn treatment strategies for hemodynamic support with <b>Reinforcement Learning (RL)</b>, with a particular focus on optimizing dosing policies for the delivery of fluids and vasopressors from historical patient data. These methods rely on compact representations of the patient's <b>state</b> to encode all information on the patient needed to inform treatment decisions; however, how to construct such representations from clinical observations of a patient has remained an open problem.
Motivated by this question, this thesis explores recent <b>self-supervised learning</b> approaches to state representation construction using deep learning. To derive informative representations of patient states from clinical measurements and past treatments of a patient, two encoding architectures are examined, a <b>Continuous Kernel Convolutional Network (CKCNN)</b> and a <b>Causal Transformer</b>, and trained on a range of self-supervised and supervised objectives, including forward dynamics modeling, behavior cloning and autoencoding, incentivizing the encoders to learn diverse, task-informative representations; the state representations provided by the encoders are then used for optimizing treatment policies for hemodynamic support with <b>deep offline reinforcement learning</b>.
Quantitative evaluation using Off-Policy Policy Evaluation on patient trajectories from the MIMIC-III and AmsterdamUMCdb intensive care databases demonstrates that treatment policies learnt using self-supervised state representations have the potential to significantly improve treatment outcomes over policies with conventional state spaces, showing substantial improvement in expected performance. Through visual inspection of the learnt policies, we observe treatment strategies similar to physicians with a slightly more restraint use of fluids and liberal use of vasopressors, in line with previous studies, and find that dosing decisions can be well explained through clinically-interpretable features extracted by the encoders; however, despite the promising results, significant differences in strategies were observed between treatment policies obtained using different state encoders, suggesting a need for caution when interpreting treatment decisions. 

<b>Keywords:</b> hemodynamic support $\cdot$ reinforcement learning $\cdot$ state representation learning $\cdot$ continuous kernel convolutional networks $\cdot$ transformers.

## Dependencies
A Python installation 3.8+ with the following libraries installed:
- numpy
- matplotlib
- pandas
- torch
- tqdm
- seaborn

#### Installation
0. (optional) we recommended to create a fresh virtual environment to ensure no clashing of library versions:
    - With python 3.8+ installed, open a command prompt
    - Navigate to repository, e.g. `cd C:/Users/<USER>/Desktop/<REPOSITORY_NAME>`
    - Call `python -m venv thesis_venv`
    - Call `thesis_venv\Scripts\activate` to activate virtual environment 
    
1. Install libraries:
    - `pip install -r requirements.txt`

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
To reproduce the results reported in the thesis, follow the following steps:

#### Preliminaries
1. Download models, preprocessed datasets and estimated behavior policies from **ADD LINK TO DRIVE!** 
    - Unpack models in `experiments/results`. You should have a folder structure as `experiments/results/<DATASET_NAME>/<EXPERIMENT_NAME>`
    - Unpack datasets in `preprocessing/datasets`
    - Unpack behavior policies for MIMIC-III/AmsterdamUMCdb in `ope/physician_policy`  

#### OPE Evaluation Results
  
1. Navigate to `experiments/plotting/ope.py`:
    - Set `model` path to the location of the model to be evaluated, e.g. `../results/amsterdam-umc-db/transformer_experiment_00000/model.pt`
    - Set `dataset_file ` path to the location of the dataset to evaluate model on, e.g. '../../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_2h/test.csv'
    - Set `behavior_policy_file ` path to location of pre-estimated behavior policy, e.g. '../../ope/physician_policy/amsterdam-umc-db_aggregated_full_cohort_2h_mlp/test_behavior_policy.csv'
  
2. Run `ope.py`
    - This might take a while as an FQE model is fit onto the action distribution of the policy network
  
For physician's policy OPE results, replace `model` by filename of behavior policy file, i.e. `model = 'behavior_policy_tmp.csv'`

#### Action Matrices

1. Navigate to `experiments/plotting/action_matrices.py`
    - Set `dataset_file ` path to the location of the dataset to evaluate model on, e.g. '../../preprocessing/datasets/amsterdam-umc-db/aggregated_full_cohort_2h/test.csv'
    - Set `dataset_label` to `mimic-iii` or `amsterdam-umc-db` (depending on the dataset chosen)
  
2. Run `action_matrices.py`


#### Input Attribution Maps

**TODO**
  

### Pretraining an Encoder

A different script was used to train each of the encoders:

**Transformer**: In `pretraining/pretrain_transformer.py` set `task` to pretraining objective of choice (either of `ae|fp|bc|mt`) and run script.<br>
**CKCNN**: In `pretraining/pretrain_ckcnn.py` set `task` to pretraining objective of choice (either of `ae|fp|bc|mt`) and run script.

### Optimizing a Treatment Policy

In `experiments/train_dqn_pretrained.py` and set paths to training and validation datasets of choice, i.e. the `train.csv` and `valid.csv` from AmsterdamUMCdb or MIMIC-III, and set `behavior_policy_file` to the behavior policy estimated for this dataset. Set `encoder` path to location of pretrained encoder and run script.
