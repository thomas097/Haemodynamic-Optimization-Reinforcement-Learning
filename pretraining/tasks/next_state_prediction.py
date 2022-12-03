import os
import torch
import math
import numpy as np
import pandas as pd

from tqdm import tqdm
from experience_replay import PrioritizedReplay
from performance_tracking import PerformanceTracker
from behavior_cloning import CosineWarmupScheduler


class EvaluationCallback:
    def __init__(self, valid_dataloader, mask_missing, n_features, num_samples=10000):
        """ Callback to evaluate model intermittently on validation set during training
        :param valid_dataloader:  DataLoader object loaded with validation set
        :param num_samples:       Max number of histories to evaluate on (for debugging purposes)
        """
        self._dataloader = valid_dataloader
        self._mse_loss = torch.nn.MSELoss()
        self._mask_missing = mask_missing
        self._num_samples = num_samples
        self._n_features = n_features

    def __call__(self, model, batch_size):
        """ Evaluates model on validation set
        :param model:       Pytorch regression model
        :param batch_size:  Number of states per batch
        :return:            MSE and Hubert scores on validation data
        """
        mse = []
        total_samples = 0

        model.eval()
        with torch.no_grad():
            # collect predictions of model for states in dataset
            self._dataloader.reset()
            while total_samples <= self._num_samples:
                total_samples += batch_size
                states, _, _, next_states, _, _ = self._dataloader.sample(batch_size, deterministic=True)

                pred_next_state = model(states)
                true_next_state = next_states[:, -1, :self._n_features]

                # mask out missing values if any
                if self._mask_missing:
                    missing_mask = 1 - states[:, -1, self._n_features:]
                    pred_next_state = pred_next_state * missing_mask
                    true_next_state = true_next_state * missing_mask

                loss = self._mse_loss(pred_next_state, true_next_state)
                mse.append(loss.item())
        model.train()

        # compute mean squared error loss
        return {'valid_mse': np.mean(mse)}


def fit_next_state(experiment,
                   encoder,
                   decoder,
                   train_data,
                   valid_data,
                   mask_missing=True,
                   batches_per_epoch=100,
                   epochs=200,
                   warmup=25,
                   lrate=1e-3,
                   batch_size=32,
                   truncate=256,
                   save_on=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('\nRunning on %s' % str(device).upper())

    tracker = PerformanceTracker(experiment)
    tracker.save_experiment_config(encoder=encoder.config, experiment=locals())

    # Optimize combined encoder-regressor model w.r.t MSE loss
    # We use a learning rate scheduler with warm-up to improve final performance,
    # see: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    model = torch.nn.Sequential(encoder, decoder).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    scheduler = CosineWarmupScheduler(warmup=warmup, max_epochs=epochs, max_lrate=lrate)
    criterion = torch.nn.MSELoss()

    # Gradient clipping to minimize influence of bad batches
    for w in model.parameters():
        if w.requires_grad:
            w.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    #####################
    #     Training      #
    #####################

    # Load training data into replay buffer (set to uniform sampling)
    train_dataloader = PrioritizedReplay(train_data, device=device, alpha=0, beta0=0, max_len=truncate)
    n_features = len(train_data.filter(regex='x\d+').columns) // 2

    # Callback to evaluate model on valid data (set to deterministic sampling!)
    valid_dataloader = PrioritizedReplay(valid_data, device=device, max_len=truncate)
    valid_callback = EvaluationCallback(valid_dataloader, mask_missing=mask_missing, n_features=n_features)

    for ep in range(epochs):
        losses = []

        # Update learning rate acc. to scheduler
        scheduler.set(optimizer=optimizer, epoch=ep)

        for _ in tqdm(range(batches_per_epoch), desc='Ep %d' % ep):
            # sample random batch of (s, s') pairs from training data
            states, _, _, next_states, _, _ = train_dataloader.sample(batch_size)

            pred_next_state = model(states)
            true_next_state = next_states[:, -1, :n_features]

            # mask out missing values if any
            if mask_missing:
                missing_mask = 1 - states[:, -1, n_features:]
                pred_next_state = pred_next_state * missing_mask
                true_next_state = true_next_state * missing_mask

            loss = criterion(pred_next_state, true_next_state)
            losses.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ep == 0 or len(losses) == batches_per_epoch:
                break

        ############################
        #   Performance Tracking   #
        ############################

        # Track training and validation loss
        tracker.add(train_mse=np.mean(losses))
        tracker.add(**valid_callback(model, batch_size=batch_size))
        tracker.save_metrics()

        print('\nEp %d/%d: %s' % (ep, epochs, tracker.print_stats()))

        # Save models upon improvement (or always if save_on=False)
        new_best = tracker.new_best(metric=save_on)
        if new_best:
            print('Model improved! Saving...')

        if not save_on or new_best:
            tracker.save_model_pt(model, 'nsp_model')
            tracker.save_model_pt(encoder, 'encoder')

    print('Done!')
