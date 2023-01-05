import copy
import os
import torch
import math
import numpy as np
import pandas as pd
from tqdm import tqdm
from experience_replay import PrioritizedReplay
from performance_tracking import PerformanceTracker
from warmup_scheduler import CosineWarmupScheduler


class EvaluationCallback:
    def __init__(self, valid_dataloader, task=None, mask_missing=False, num_samples=10000):
        """ Callback to evaluate model intermittently on validation set during training
        :param valid_dataloader:  DataLoader object loaded with validation set
        :param num_samples:       Max number of histories to evaluate on (for debugging purposes)
        """
        self._dataloader = valid_dataloader
        self._mask_missing = mask_missing
        self._criterion = TaskLoss(task=task)
        self._num_samples = num_samples

    def __call__(self, encoder, decoder_ae, decoder_bc, decoder_fp, batch_size):
        """ Evaluates model on validation set
        :param model:       Pytorch regression model
        :param batch_size:  Number of states per batch
        :return:            MSE and Hubert scores on validation data
        """
        mse = []
        total_samples = 0

        with torch.no_grad():
            # collect predictions of model for states in dataset
            self._dataloader.reset()
            while total_samples <= self._num_samples:
                total_samples += batch_size
                obs, _, _, next_obs, _, _ = self._dataloader.sample(batch_size, deterministic=True)

                # If masking is enabled, mask out imputed values from reconstruction loss
                mask = None
                if self._mask_missing:
                    with torch.no_grad():
                        mask = torch.absolute(obs[:, -1] - next_obs[:, -1]) > 1e-5  # any difference between variables of observations?

                # predictions
                state = encoder(obs)
                action = next_obs[:, -1, :2]
                pred_obs = decoder_ae(state)
                pred_next_obs = decoder_fp(torch.concat([state, action], dim=1))
                pred_action = decoder_bc(state)

                # targets
                true_obs = obs[:, -1]
                true_next_obs = next_obs[:, -1, 2:]
                true_action = next_obs[:, -1, :2]  # note: the current action was concatenated to the next observation!

                loss = self._criterion(pred_obs, pred_next_obs, pred_action, true_obs, true_next_obs, true_action, mask=mask)
                mse.append(loss.item())

        # compute mean squared error loss
        return {'valid_mse': np.mean(mse)}


class TaskLoss:
    def __init__(self, task):
        assert task in ['mt', 'fp', 'ae', 'bc']
        self._task = task
        self._mse = torch.nn.MSELoss()

    def __call__(self, pred_obs, pred_next_obs, pred_action, true_obs, true_next_obs, true_action, mask=None):
        """ Computes loss w.r.t. pretext learning objective (incl. autoencoding ('ae'),
        behavior cloning ('bc'), forward modeling ('fm') or multi-task ('mt'))
        """
        # If masking is enabled, mask imputed values in observation reconstruction loss
        if mask is not None:
            pred_obs = mask * pred_obs
            pred_next_obs = mask[:, 2:] * pred_next_obs
            true_obs = mask * true_obs
            true_next_obs = mask[:, 2:] * true_next_obs

        # Compute loss w.r.t. pretext task
        if self._task == 'mt':
            return self._mse(pred_obs, true_obs) + self._mse(pred_next_obs, true_next_obs) + self._mse(pred_action, true_action) / 3
        elif self._task == 'ae':
            return self._mse(pred_obs, true_obs)
        elif self._task == 'fp':
            return self._mse(pred_next_obs, true_next_obs)
        elif self._task == 'bc':
            return self._mse(pred_action, true_action)


def create_decoder(in_channels, out_channels):
    # decoder maps from encoder output back to input channels!
    decoder = torch.nn.Sequential(
        torch.nn.Linear(out_channels, 128),
        torch.nn.LeakyReLU(),
        torch.nn.Linear(128, in_channels)
    )
    return decoder


def fit_multi_task(experiment,
                   task,
                   encoder,
                   in_channels,
                   out_channels,
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

    # Create decoders needed for tasks
    decoder_ae = create_decoder(in_channels, out_channels)
    decoder_bc = create_decoder(2, out_channels)
    decoder_fp = create_decoder(in_channels - 2, out_channels + 2)

    tracker = PerformanceTracker(experiment)
    #tracker.save_experiment_config(encoder=encoder.config, experiment=locals())

    # Optimize combined encoder-decoder model using task-specific MSE loss
    # We use a learning rate scheduler with warm-up to improve final performance,
    # see: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    model = torch.nn.ModuleList([encoder, decoder_ae, decoder_bc, decoder_fp]).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    scheduler = CosineWarmupScheduler(warmup=warmup, max_epochs=epochs, max_lrate=lrate)
    criterion = TaskLoss(task=task)

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
    valid_callback = EvaluationCallback(valid_dataloader, task=task, mask_missing=True)

    for ep in range(epochs):
        losses = []

        # Update learning rate acc. to scheduler
        scheduler.set(optimizer=optimizer, epoch=ep)

        for _ in tqdm(range(batches_per_epoch), desc='Ep %d' % ep):
            # sample random batch of (Ot, Ot+1) pairs from training data
            obs, _, _, next_obs, _, _ = train_dataloader.sample(batch_size) # obs includes previous actions!

            # If masking is enabled, mask out imputed values from reconstruction loss
            mask = None
            if mask_missing:
                with torch.no_grad():
                    mask = torch.absolute(obs[:, -1] - next_obs[:, -1]) > 1e-5  # any difference between variables of observations?

            # predictions
            state = encoder(obs)
            action = next_obs[:, -1, :2]
            pred_obs = decoder_ae(state)
            pred_next_obs = decoder_fp(torch.concat([state, action], dim=1))
            pred_action = decoder_bc(state)

            # targets
            true_obs = obs[:, -1]
            true_next_obs = next_obs[:, -1, 2:] # all but the actions
            true_action = next_obs[:, -1, :2]   # note: the action was concatenated to the next observation!

            loss = criterion(pred_obs, pred_next_obs, pred_action, true_obs, true_next_obs, true_action, mask=mask)
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
        tracker.add(**valid_callback(encoder, decoder_ae, decoder_bc, decoder_fp, batch_size=64))
        tracker.save_metrics()

        print('\nEp %d/%d: %s' % (ep, epochs, tracker.print_stats()))

        # Save models upon improvement (or always if save_on=False)
        new_best = tracker.new_best(metric=save_on, maximize=False)
        if new_best:
            print('Model improved! Saving...')

        if not save_on or new_best:
            autoencoder_model = torch.nn.Sequential(encoder, decoder_ae)
            forward_model = torch.nn.Sequential(encoder, decoder_fp)
            tracker.save_model_pt(autoencoder_model, 'autoencoder_model')
            tracker.save_model_pt(forward_model, 'forward_model')
            tracker.save_model_pt(encoder, 'encoder')

    print('Done!')
