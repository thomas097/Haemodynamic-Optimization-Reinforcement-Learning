# Reinforcement Learning with Self-Supervised State Representations for Hemodynamic Support at the ICU

This repository contains the source code accompanying the MSc thesis <i>"Reinforcement learning with self-supervised state representations for hemodynamic support at the ICU"</i> in partial fulfillment of the degree of MSc Artificial Intelligence at the Vrije Universiteit (VU), Amsterdam.

<img src="https://github.com/thomas097/Haemodynamic-Optimization-Reinforcement-Learning/blob/main/images/abstract.PNG?raw=true"  width="50%" height="50%">

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
