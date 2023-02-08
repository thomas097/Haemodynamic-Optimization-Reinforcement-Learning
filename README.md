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

- `exploratory_analysis/`: Exploratory Data Analysis (EDA) of the MIMIC-III and AmsterdamUMCdb intensive care databases

- `data/`: this folder contains all *data extraction* source code used to extract raw unprocessed patient data from MIMIC-III and the AmsterdamUMCdb part of the input (i.e. observation- and action-space), including vitals, lab results, demographics, treatment parameters, discharge information, etc.

- `preprocessing/`: Jupyter notebooks used to process and convert the raw measrurements and treatment parameters extracted from MIMIC-III and the AmsterdamUMCdb into regularly-sampled patient trajectories with 2h time steps.

- `model/`: Implementations of the CKCNN and Transformer encoders and a range of baseline state space encoders (e.g. last observation, last K observatons, LSTM, etc.). This folder also contains the implementation of Dueling Double Deep Q-Learning (D3QN), used to train treatment policies. For details, see the report. 

- `pretraining/`: Implementation and training routines of the autoencoding, forward dynamics modeling, behavior cloning and the combined 'multi-task' pretraining tasks to (pre)train the encoders for state extraction.

- `experiments/`: Training routines to optimize treatment policies on MIMIC-III or the AmsterdamUMCdb with a (pre)trained encoder. Also, contains evaluation routines to evaluate the resulting treatment policies with OPE and generate action matrices (see instructions below).

- `ope/`: Definitions of the Off-policy policy evaluation estimators (PHWIS, PHWDR and FQE) used to evaluate the learnt policies, and routines used to estimate the physician's behavior policy.

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
py -3 ope.py --dataset <dataset> --model <model> --partition <partition>
```
  
- `--dataset`: which dataset to use to evaluate policy (`mimic-iii|amsterdam-umc-db`). As policies are traint on the training partition of the same dataset, the correct policy network is chosen automatically.
- `--model`: which encoder-policy pair to evaluate (`last_state|concat-2|concat-3|autoencoder|lstm_mt|ckcnn|transformer|physician`)
- `--partition`: which partition of the dataset to use for evaluation (`valid|test`).

Note: `last_state` refers to the _Handcrafted State_ in the report

#### Action Matrices

```
py -3 action_matrices.py --dataset <dataset> --models <models> --partition <partition>
```
  
- `--dataset`: which dataset to use to evaluate policy (`mimic-iii|amsterdam-umc-db`).
- `--models`: which encoder-policy pairs to evaluate (`last_state|concat-2|concat-3|autoencoder|lstm_mt|ckcnn|transformer`). You may fill in multiple policies, e.g. `--models last_state transformer ckcnn`
- `--partition`: which partition of the dataset to use for evaluation (`valid|test`).


#### Input Attribution Maps

**TODO**
  

### (Pre)Training a State Encoder

To simplify the training of the encoders (CKCNN, Transformer and LSTM baseline), separate scripts are used.

To train a Transformer encoder (with the parameters as used in the report), run:
```
py -3 pretrain_transformer.py --dataset <dataset> --task <task> --out_dims <out_dims> --lrate <lrate> --epochs <epochs> --batches_per_epoch <batches_per_epoch> --warmup <warmup> --batch_size <batch_size>
```

To train a CKCNN encoder, run:
```
py -3 pretrain_ckcnn.py --dataset <dataset> --task <task> --out_dims <out_dims> --lrate <lrate> --epochs <epochs> --batches_per_epoch <batches_per_epoch> --warmup <warmup> --batch_size <batch_size>
```

To train an LSTM encoder baseline (choose `--task mt` for same results), run:
```
py -3 pretrain_baselines.py --dataset <dataset> --task <task> --out_dims <out_dims> --lrate <lrate> --epochs <epochs> --batches_per_epoch <batches_per_epoch> --warmup <warmup> --batch_size <batch_size>
```

- `--dataset`: which dataset to use to train encoder (`mimic-iii|amsterdam-umc-db`).
- `--task`: which learning objective to use (`fp|bc|ae|mt`).
- `--out_dims`: Output dimensionality of the state (default: `96`).
- `--lrate`: Learning rate (default: `5e-4`).
- `--epochs`: Number of training iterations (default: `200`).
- `--batches_per_epoch`: Number of batches of histories to sample that together form one interation (default: `500`).
- `--warmup`: Number of linear warm-up steps to gradually increase learning rate (default: `50`). This parameter is particularly importantant for the Transformer as decreasing this number below 40 steps was found to decrease likelihood of convergence.
- `--batch_size`: Number of histories sampled in each batch (default: `32`).


### Optimizing a Treatment Policy on MIMIC-III / AmsterdamUMCdb

Separate scripts are used to optimize the pretrained encoders and baselines:

To train a value network (policy) on top of a pretrained encoder (e.g. CKCNN), run:
```
py -3 train_dqn_pretrained.py --dataset <dataset> --model <model> --out_dims <out_dims> --alpha <alpha> --gamma <gamma> --tau <tau> --episodes <episodes> --eval_after <eval_after> --freeze_encoder <freeze_encoder> --replay_alpha <replay_alpha> --replay_beta <replay_beta> --batch_size <batch_size>
```

To train a policy using a baseline state space (e.g. handcrafted state space), run:
```
py -3 train_dqn_baselines.py --dataset <dataset> --baseline <baseline> --out_dims <out_dims> --alpha <alpha> --gamma <gamma> --tau <tau> --episodes <episodes> --eval_after <eval_after> --freeze_encoder <freeze_encoder> --replay_alpha <replay_alpha> --replay_beta <replay_beta> --batch_size <batch_size>
```
Using the following arguments:

- `--dataset`: which dataset to use to train encoder (`mimic-iii|amsterdam-umc-db`).
- `--model`: which pretrained encoder to use (e.g.\ `transformer_mt` when a transformer was trained using the MT pretraining objective).
- `--baseline`: when no pretrained encoder is used, which baseline state space to employ instead (`last_state|concat-2|concat-3|autoencoder|tcn|lstm|gru`). **Note:** In case LSTM, GRU or TCN is selected, no pretraining is applied; therefore, the models are trained __from scratch__! 
- `--out_dims`: Output dimensionality of the encoder's state (default: `96`).
- `--alpha`: Learning rate of main value network (default: `1e-4`).
- `--gamma`: Discount factor (default: `0.95`).
- `--tau`: Learning rate of target network (default: `1e-4`).
- `--episodes`: Number of training episodes (default: `200000`).
- `--eval_after`: After how many episodes to call an evaluation callback for an intermittant evaluation with OPE (shown on screen) (default: `5000`).
- `--freeze_encoder`: If encoder has been pretrained, whether to continue fine-tuning (`False`) or freeze its weights (`True`) (default: `False`).
- `--replay_alpha/beta`: Parameters of the prioritized experience replay buffer (default: `replay_alpha=0.6` and `replay_beta=0.9`). For details, see [PER paper](https://arxiv.org/abs/1511.05952) on Arxiv.
- `--batch_size`: Number of transitions sampled for each update (default: `32`).
