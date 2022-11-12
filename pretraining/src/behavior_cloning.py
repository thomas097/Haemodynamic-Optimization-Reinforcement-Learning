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
    def __init__(self, valid_df, device, maxlen=0, num_samples=20):
        """
        Callback to evaluate model intermittently on validation set during training
        :param valid_df:     DataFrame with validation set
        :param device:       Torch device on which model is stored
        :param maxlen:       Maximum length of history
        :param num_samples:  Max number of histories to evaluate on (for debugging purposes)
        """
        self._data = DataLoader(valid_df, maxlen=maxlen, device=device)
        self._num_samples = num_samples

    @staticmethod
    def _cross_entropy(y_pred, y_true, weights):
        """
        Class-weighted cross entropy loss
        :param y_pred:   Model predictions
        :param y_true:   Target labels
        :param weights:  Tensor containing a weight for each class
        :return:         Loss value
        """
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        return criterion(y_pred, y_true).item()

    @staticmethod
    def _f1_score(y_pred, y_true):
        """
        Macro-average F1 score
        :param y_pred:   Model predictions
        :param y_true:   Target labels
        :return:         F1 score
        """
        y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        return f1_score(y_true, y_pred, average='macro')

    def __call__(self, model, batch_size, weights=None):
        """
        Evaluates model on validation set
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
        scores = {'valid_loss': self._cross_entropy(y_pred, y_true, weights),
                  'valid_f1': self._f1_score(y_pred, y_true)}
        return scores


class CosineWarmupScheduler:
    def __init__(self, warmup, max_lrate, max_epochs):
        """
        Cosine-based learning rate regime with warm-up
        :param warmup:      Number of warm-up epochs
        :param max_lrate:   Maximum learning rate
        :param max_epochs:  Number of epochs
        """
        print('\nUsing CosineWarmupScheduler')
        self._warmup = warmup
        self._max_lrate = max_lrate
        self._max_epochs = max_epochs

    def set(self, optimizer, epoch):
        """
        Set optimizer learning rate according to scheduler
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


def get_class_weights(df):
    """
    Weigh action classes inversely proportional to their frequency
    :param df:  DataFrame with training data
    :return:    Tensor of class weights
    """
    print('\nComputing action class weights')
    # Compute frequencies of actions relative to most freq class
    counts = df.action.value_counts()
    weights = torch.zeros(25)
    for i, count in counts.to_dict().items():
        weights[int(i)] = (1 / count) * (counts.sum() / 25)

    print('Action class weights:')
    print(weights, '\n')
    return weights.detach()


def oversample_vasopressors(train_df, max_oversample):
    """
    Oversample episodes with infrequent actions (usually vasopressors)
    to account for action class imbalance in training data
    :param df:               DataFrame containing training data
    :param max_oversample:   Maximum number of episodes to add to existing dataset
    :return:                 Augmented training set DataFrame with duplicated episodes
    """
    print('\nOversampling trajectories with infrequent VP doses')
    # Count frequency of actions
    action_counts = train_df.action.value_counts(dropna=True)

    # Compute oversampling rate as log(count(most_freq_action) / count(action))
    oversampling_rates = (action_counts.max() / action_counts).apply(lambda x: math.log(x, 1.1))

    # Identify actions with the most infrequent actions (on average)
    episode_rates = []
    for _, episode in train_df.groupby('episode', sort=False):
        # Compute avg. oversampling rate for actions in episode
        avg_rate = int(np.mean([oversampling_rates.loc[a] for a in episode.action if not np.isnan(a)]))
        episode_rates.append((avg_rate, episode))

    # Repeat episodes with the most rare actions proportional to their estimated rate
    # while collectively remaining below max_oversample
    max_episode_id = train_df.episode.astype(int).max()
    new_episodes = []
    for rate, episode in sorted(episode_rates, key=lambda x: -x[0]):
        if max_oversample - rate > 0:
            max_oversample -= rate

            # Copy episodes proportional to their rate
            # and assign new episode id
            for _ in range(rate):
                max_episode_id += 1
                new_episode = episode.copy()
                new_episode.episode = max_episode_id
                new_episodes.append(new_episode)

    # Augment original dataset
    new_train_df = pd.concat([train_df] + new_episodes, axis=0).reset_index(drop=True)
    print('Total episodes: %d' % len(new_train_df.episode.unique()))
    return new_train_df


def fit_behavior_cloning(experiment,
                         encoder,
                         classifier,
                         train_data,
                         valid_data,
                         batches_per_epoch=100,
                         oversample_vaso=0,
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

    # Oversample vasopressors to counteract action imbalance in training set
    if oversample_vaso > 0:
        train_data = oversample_vasopressors(train_data, max_oversample=oversample_vaso)

    # Optimize combined encoder-classifier model w.r.t class-weighted CE loss
    # We use a learning rate scheduler with warm-up to improve final performance,
    # see: https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
    model = torch.nn.Sequential(encoder, classifier).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    scheduler = CosineWarmupScheduler(warmup=warmup, max_epochs=epochs, max_lrate=lrate)

    class_weights = get_class_weights(train_data).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights)

    # Gradient clipping to minimize influence of bad batches
    for w in model.parameters():
        if w.requires_grad:
            w.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    #####################
    #     Training      #
    #####################

    # Load training data into replay buffer (set to random sampling)
    train_dataloader = DataLoader(train_data, maxlen=truncate, device=device)

    # Callback to evaluate model on valid data
    valid_callback = EvaluationCallback(valid_data, maxlen=truncate, device=device)

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
        tracker.add(**valid_callback(model, batch_size=batch_size, weights=class_weights))
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
