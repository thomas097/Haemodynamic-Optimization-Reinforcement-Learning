# Reinforcement Learning with Self-Supervised State Representations for Hemodynamic Support at the ICU

This repository contains the source code accompanying the MSc thesis <i>"Reinforcement learning with self-supervised state representations for hemodynamic support at the ICU"</i> in partial fulfillment of the degree of MSc Artificial Intelligence at the Vrije Universiteit (VU), Amsterdam.

<img src="https://github.com/thomas097/Haemodynamic-Optimization-Reinforcement-Learning/blob/main/images/abstract.PNG?raw=true"  width="50%" height="50%">

## Dependencies
A Python installation 3.8+ with the following libraries installed:
- matplotlib == 3.6.0
- matplotlib-inline == 0.1.6
- numpy == 1.23.3
- pandas == 1.5.0
- scikit-learn == 1.1.2
- scipy == 1.9.2
- seaborn == 0.12.0
- sklearn == 0.0
- torch == 1.13.1
- tqdm == 4.64.1

#### Installation
0. (optional) we recommended to create a fresh virtual environment to ensure no conflicts between library versions:
    - With python 3.8+ installed, open a command prompt
    - Navigate to repository by `cd <PATH_TO_REPOSITORY>`
    - Call `python -m venv thesis_venv`, followed by `thesis_venv\Scripts\activate` to activate virtual environment 
    
1. Install libraries:
    - `pip install -r requirements.txt`

## Structure

The repository is organized as follows:

- `exploratory_analysis`: Exploratory Data Analysis (EDA) of the MIMIC-III and AmsterdamUMCdb intensive care databases

- `/data`: this folder contains all *data extraction* source code used to extract raw unprocessed patient data from MIMIC-III and the AmsterdamUMCdb part of the input (i.e. observation- and action-space), including vitals, lab results, demographics, treatment parameters, discharge information, etc.

- `preprocessing`: Jupyter notebooks used to process and convert the raw measrurements and treatment parameters extracted from MIMIC-III and the AmsterdamUMCdb into regularly-sampled patient trajectories with 2h time steps.

- `model`: Implementations of the CKCNN and Transformer encoders and a range of baseline state space encoders (e.g. last observation, last K observatons, LSTM, etc.). This folder also contains the implementation of Dueling Double Deep Q-Learning (D3QN), used to train treatment policies. For details, see the report. 

- `pretraining`: Implementation and training routines of the autoencoding, forward dynamics modeling, behavior cloning and the combined 'multi-task' pretraining tasks to (pre)train the encoders for state extraction.

- `experiments`: Training routines to optimize treatment policies on MIMIC-III or the AmsterdamUMCdb with a (pre)trained encoder. Also, contains evaluation routines to evaluate the resulting treatment policies with OPE and generate action matrices (see instructions below).

- `ope`: Definitions of the Off-policy policy evaluation estimators (PHWIS, PHWDR and FQE) used to evaluate the learnt policies, and routines used to estimate the physician's behavior policy.

## Instructions

### Reproducing the Reported Results
To reproduce the results reported in the thesis, you may follow the instructions below:

#### Preliminaries
First, download and unpack zip files of models, preprocessed datasets and estimated behavior policies from **ADD LINK TO DRIVE!** (**TODO: check whether this is legally allowed**). You should obtain a file structure as follows:

    .
    ├── ...
    ├── experiments
    │   └── results
    │       └── amsterdam-umc-db
    |       │   ├── ...
    │       │   ├── concat-2_experiment_00000
    │       │   │   └── <DATA_FILES>
    │       │   ├── transformer_experiment_00000
    │       │   │   └── <DATA_FILES>
    │       │   └── ckcnn_experiment_00000
    │       │       └── <DATA_FILES>
    │       └── mimic-iii
    |           ├── ...
    │           ├── concat-2_experiment_00000
    │           │   └── <DATA_FILES>
    │           ├── transformer_experiment_00000
    │           │   └── <DATA_FILES>
    │           └── ckcnn_experiment_00000
    │               └── <DATA_FILES>
    ├── preprocessing
    │   └── datasets
    │       └── amsterdam-umc-db
    │       │   └── aggregated_full_cohort_2h
    │       │       └── <DATA_FILES>
    │       └── mimic-iii
    │           └── aggregated_full_cohort_2h
    │               └── <DATA_FILES>
    ├── ope
    │   └── physician_policy
    │       └── amsterdam-umc-db_aggregated_full_cohort_2h_mlp
    │       │   └── <DATA_FILES>
    │       └── mimic-iii_aggregated_full_cohort_2h_mlp
    │           └── <DATA_FILES>
    └── ...

#### Off-policy Policy Evaluation (OPE)

```
py -3 ope.py --dataset <DATASET> --model <MODEL> --partition <PARTITION>
```
  
- DATASET = `mimic-iii|amsterdam-umc-db`
- MODEL = `last_state|concat-2|concat-3|autoencoder|lstm_mt|ckcnn|transformer|physician`
- PARTITION = `valid|test`

Note: `last_state` denotes the _Handcrafted State_ in the report

#### Action Matrices

```
py -3 action_matrices.py --dataset <DATASET> --models <MODELS> --partition <PARTITION>
```
  
- DATASET = `mimic-iii|amsterdam-umc-db`
- MODELS = any combination of `last_state|concat-2|concat-3|autoencoder|lstm_mt|ckcnn|transformer`, e.g. '--models transformer ckcnn'
- PARTITION = `valid|test`


#### Input Attribution Maps

**TODO**
  

### Pretraining an Encoder

A different script was used to train each of the encoders:

**Transformer**: In `pretraining/pretrain_transformer.py` set `task` to pretraining objective of choice (either of `ae|fp|bc|mt`) and run script.<br>
**CKCNN**: In `pretraining/pretrain_ckcnn.py` set `task` to pretraining objective of choice (either of `ae|fp|bc|mt`) and run script.

### Optimizing a Treatment Policy

In `experiments/train_dqn_pretrained.py` and set paths to training and validation datasets of choice, i.e. the `train.csv` and `valid.csv` from AmsterdamUMCdb or MIMIC-III, and set `behavior_policy_file` to the behavior policy estimated for this dataset. Set `encoder` path to location of pretrained encoder and run script.
