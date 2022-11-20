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
    def __init__(self, valid_dataloader, num_samples=5000):
        """ Callback to evaluate model intermittently on validation set during training
        :param valid_dataloader:  DataLoader object loaded with validation set
        :param num_samples:       Max number of histories to evaluate on (for debugging purposes)
        """
        self._dataloader = valid_dataloader
        self._mse = torch.nn.MSELoss()
        self._hubert = torch.nn.HuberLoss()
        self._num_samples = num_samples

    def __call__(self, model, batch_size):
        """
        Evaluates model on validation set
        :param model:       Pytorch regression model
        :param batch_size:  Number of states per batch
        :return:            MSE and Hubert scores on validation data
        """
        # Collect predictions of model and target labels
        y_pred, y_true = [], []

        model.eval()
        with torch.no_grad():
            total_samples = 0

            # Collect predictions of model for states in dataset
            while total_samples <= self._num_samples:
                total_samples += batch_size
                states, _, _, next_states, _, _ = self._dataloader.sample(batch_size, deterministic=True)

                y_pred.append(model(states).detach())
                y_true.append(next_states[:, -1])
        model.train()

        y_pred = torch.concat(y_pred, dim=0)
        y_true = torch.concat(y_true, dim=0)

        # Compute cross entropy loss and F1 scores
        scores = {'hubert_loss': self._hubert(y_pred, y_true).item(),
                  'mse_loss': self._mse(y_pred, y_true).item()}
        return scores


def fit_next_state(experiment,
                   encoder,
                   decoder,
                   train_data,
                   valid_data,
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

    # Optimize combined encoder-regressor model w.r.t Hubert loss
    # We use a learning rate scheduler with warm-up to improve final performance,
    # see: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    model = torch.nn.Sequential(encoder, decoder).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    scheduler = CosineWarmupScheduler(warmup=warmup, max_epochs=epochs, max_lrate=lrate)
    criterion = torch.nn.HuberLoss()

    # Gradient clipping to minimize influence of bad batches
    for w in model.parameters():
        if w.requires_grad:
            w.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    #####################
    #     Training      #
    #####################

    # Load training data into replay buffer (set to uniform sampling)
    train_dataloader = PrioritizedReplay(train_data, device=device, alpha=0, beta0=0, max_len=truncate)

    # Callback to evaluate model on valid data (set to deterministic sampling!)
    valid_dataloader = PrioritizedReplay(valid_data, device=device, max_len=truncate)
    valid_callback = EvaluationCallback(valid_dataloader)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        total_batches = 0

        # Update learning rate acc. to scheduler
        scheduler.set(optimizer=optimizer, epoch=ep)

        for _ in tqdm(range(batches_per_epoch), desc='Ep %d' % ep):
            # Sample random batch of (s, s') pairs from training data
            states, _, _, next_states, _, _ = train_dataloader.sample(batch_size)

            # Compute error of model
            loss = criterion(model(states), next_states[:, -1]) # drop history of next state!
            total_loss += loss.item()
            total_batches += 1

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ep == 0 or total_batches == batches_per_epoch:
                break

        ############################
        #   Performance Tracking   #
        ############################

        # Track training and validation loss
        tracker.add(train_loss=total_loss / total_batches)
        tracker.add(**valid_callback(model, batch_size=batch_size))
        tracker.save_metrics()

        print('\nEp %d/%d: %s' % (ep, epochs, tracker.print_stats()))

        # Save models upon improvement (or always if save_on=False)
        new_best = tracker.new_best(metric=save_on)
        if new_best:
            print('Model improved! Saving...')

        if not save_on or new_best:
            tracker.save_model_pt(model, 'regressor')
            tracker.save_model_pt(encoder, 'encoder')

    print('Done!')
