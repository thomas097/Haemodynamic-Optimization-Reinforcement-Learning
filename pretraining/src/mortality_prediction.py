import os
import torch
import math
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import f1_score
from dataloader import DataLoader
from performance_tracking import PerformanceTracker


class EvaluationCallback:
    def __init__(self, valid_dataloader, weights=None, num_samples=500):
        """
        Callback to evaluate model intermittently on validation set during training
        :param valid_dataloader:  DataLoader object loaded with validation set
        :param num_samples:       Max number of histories to evaluate on (for debugging purposes)
        """
        self._data = valid_dataloader
        self._loss = BCELoss(weight=weights)
        self._num_samples = num_samples

    @staticmethod
    def _f1_score(y_pred, y_true):
        """ Macro-average F1 score
        :param y_pred:   Soft class predictions
        :param y_true:   Target labels
        :return:         F1 score
        """
        y_pred = (y_pred > 0.5).cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        return f1_score(y_true, y_pred)

    def __call__(self, model, batch_size, weights=None):
        """ Evaluates model on validation set
        :param model:       Pytorch classification model
        :param batch_size:  Number of samples per batch
        :param weights:     Class weights used to counteract the class imbalance (optional)
        :return:            Dict with CE loss and F1 scores on validation data
        """
        # Collect predictions of model and target labels
        y_pred, y_true = [], []

        model.eval()
        with torch.no_grad():
            total_samples = 0

            # Collect predictions of model for states in dataset
            for states, actions in self._data.iterate(batch_size, shuffle=False):
                y_pred.append(model(states).detach())
                y_true.append(actions)

                # Limit predictions to N batches
                total_samples += batch_size
                if total_samples > self._num_samples:
                    break
        model.train()

        y_pred = torch.concat(y_pred, dim=0)
        y_true = torch.concat(y_true, dim=0)

        # Compute cross entropy loss and F1 scores
        scores = {'valid_loss': self._loss(y_pred, y_true).item(),
                  'valid_f1': self._f1_score(y_pred, y_true)}
        return scores


class BCELoss:
    def __init__(self, weight):
        """ Binary Cross Entropy loss with class weights
        """
        self._weight = weight

    def __call__(self, y_pred, y_true):
        """ Computes loss of soft class predictions y_pred w.r.t. target y_true
        :param y_pred:  Positive class probability
        :param y_true:  Target label
        :returns:       Scalar CE loss
        """
        if self._weight is not None:
            assert len(self._weight) == 2
            w0, w1 = self._weight
            loss = w1 * (y_true * torch.log(y_pred)) + w0 * ((1 - y_true) * torch.log(1 - y_pred))
        else:
            loss = (y_true * torch.log(y_pred)) + ((1 - y_true) * torch.log(1 - y_pred))

        return torch.neg(torch.mean(loss))



class CosineWarmupScheduler:
    def __init__(self, warmup, max_lrate, max_epochs):
        """ Cosine-based learning rate regime with linear warm-up
        :param warmup:      Number of warm-up epochs
        :param max_lrate:   Maximum learning rate
        :param max_epochs:  Number of epochs
        """
        print('\nUsing CosineWarmupScheduler')
        self._warmup = warmup
        self._max_lrate = max_lrate
        self._max_epochs = max_epochs

    def set(self, optimizer, epoch):
        """ Set optimizer learning rate according to scheduler
        :param optimizer:  Torch Optimizer instance
        :param epoch:      Epoch number <= max_epochs
        :return:
        """
        if epoch < self._warmup:
            lrate = (epoch / self._warmup) * self._max_lrate
        else:
            lrate = self._max_lrate * 0.5 * (1 + np.cos(np.pi * (epoch - self._warmup) / (self._max_epochs - self._warmup)))

        for g in optimizer.param_groups:
            g['lr'] = lrate


def fit_mortality_prediction(experiment,
                             encoder,
                             classifier,
                             train_data,
                             valid_data,
                             batches_per_epoch=100,
                             class_weights=(0, 3),
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

    # Optimize combined encoder-classifier model w.r.t class-weighted binary CE loss
    # We use a learning rate scheduler with warm-up to improve final performance,
    # see: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    model = torch.nn.Sequential(encoder, classifier).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    scheduler = CosineWarmupScheduler(warmup=warmup, max_epochs=epochs, max_lrate=lrate)
    criterion = BCELoss(weight=torch.tensor(class_weights).to(device))

    # Gradient clipping to minimize influence of bad batches
    for w in model.parameters():
        if w.requires_grad:
            w.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    # create expire_flag label with 0 = survival and 1 = death
    def expire_flag(rewards):
        return np.ones(len(rewards)) - int(rewards.max() > 0)

    train_data['expire_flag'] = train_data.groupby('episode').reward.transform(lambda rew: expire_flag(rew))
    valid_data['expire_flag'] = valid_data.groupby('episode').reward.transform(lambda rew: expire_flag(rew))

    #####################
    #     Training      #
    #####################

    # Load training data into replay buffer (set to random sampling)
    train_dataloader = DataLoader(train_data, label_col='expire_flag', maxlen=truncate, device=device)

    # Callback to evaluate model on valid data
    valid_dataloader = DataLoader(valid_data, label_col='expire_flag', maxlen=truncate, device=device)
    valid_callback = EvaluationCallback(valid_dataloader, weights=class_weights)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        total_batches = 0

        # Update learning rate acc. to scheduler
        scheduler.set(optimizer=optimizer, epoch=ep)

        with tqdm(total=batches_per_epoch, desc='Ep %d' % ep) as pbar:
            # Sample batch of (s, a) pairs from training data
            for states, actions in train_dataloader.iterate(batch_size, shuffle=True):
                # Compute error of model
                loss = criterion(model(states), actions.flatten())
                total_loss += loss.item()
                total_batches += 1

                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if ep == 0 or total_batches == batches_per_epoch:
                    break

                pbar.update(1)

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
            tracker.save_model_pt(model, 'classifier')
            tracker.save_model_pt(encoder, 'encoder')

    print('Done!')
