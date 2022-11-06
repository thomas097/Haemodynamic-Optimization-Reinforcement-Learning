import os
import torch
import math
import numpy as np
import pandas as pd

from tqdm import tqdm
from sklearn.metrics import f1_score
from experience_replay import PrioritizedReplay
from dataloader import DataLoader
from performance_tracking import PerformanceTracker


class EvaluationCallback:
    """ Callback to evaluate model's loss on validation set
    """
    def __init__(self, valid_df, maxlen=0, num_samples=200):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self._data = DataLoader(valid_df, maxlen=maxlen, device=device)
        self._num_samples = num_samples

    @staticmethod
    def _cross_entropy(y_pred, y_true, weights):
        criterion = torch.nn.CrossEntropyLoss(weight=weights)
        return criterion(y_pred, y_true).item()

    @staticmethod
    def _f1_score(y_pred, y_true):
        y_pred = torch.argmax(y_pred, dim=1).cpu().detach().numpy()
        y_true = y_true.cpu().detach().numpy()
        return f1_score(y_true, y_pred, average='macro')

    def __call__(self, model, batch_size, weights=None):
        """ Runs evaluation over evaluation dataset """
        # Collect predictions of model and target labels
        y_pred, y_true = [], []

        model.eval()
        with torch.no_grad():
            # Collect predictions of model for states in dataset
            for k, (states, actions) in enumerate(self._data.iterate(batch_size)):
                y_pred.append(model(states).detach())
                y_true.append(actions)

                # Limit predictions to N batches
                if k > self._num_samples:
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
        self._warmup = warmup
        self._max_lrate = max_lrate
        self._max_epochs = max_epochs

    def set(self, optimizer, epoch):
        """
        Set optimizer learning rate according to scheduler

        :param optimizer:  Torch Optimizer instance
        :param epoch:     Epoch number <= max_epochs
        :return:
        """
        if epoch < self._warmup:
            lrate = (epoch / self._warmup) * self._max_lrate
        else:
            lrate = self._max_lrate * 0.5 * (1 + np.cos(np.pi * epoch / self._max_epochs))

        for g in optimizer.param_groups:
            g['lr'] = lrate


def get_class_weights(df):
    """ Weigh action classes inversely proportional to their frequency """
    counts = df.action.value_counts()
    weights = torch.zeros(25)
    for i, count in counts.to_dict().items():
        weights[int(i)] = (1 / count) * (counts.sum() / 25)

    print('Action class weights:')
    print(weights, '\n')
    return weights.detach()


def oversample_vasopressors(df, max_oversample):
    """ Oversample episodes with infrequent actions (usually vasopressors)
        to account for action class imbalance in training data
    """
    # Count frequency of actions
    action_counts = df.action.value_counts(dropna=True)

    # Compute oversampling rate as log2(count(most_freq_action) / count(action))
    oversampling_rates = (action_counts.max() / action_counts).apply(lambda x: math.log(x, 1.2))

    # Identify actions with the most infrequent actions (on average)
    episode_rates = []
    for _, episode in df.groupby('episode', sort=False):
        # Compute avg. oversampling rate for actions in episode
        avg_rate = int(np.mean([oversampling_rates.loc[a] for a in episode.action if not np.isnan(a)]))
        episode_rates.append((avg_rate, episode))

    # Repeat episodes with the most rare actions proportional to their estimated rate
    # while collectively remaining below max_oversample
    max_episode_id = df.episode.astype(int).max()
    new_episodes = []
    for rate, episode in sorted(episode_rates, key=lambda x: -x[0]):
        if max_oversample - rate > 0:
            max_oversample -= rate

            # Copy episode and assign new episode id
            # Assign each new episode a new icustay_id
            for _ in range(rate):
                max_episode_id += 1
                new_episode = episode.copy()
                new_episode.episode = max_episode_id
                new_episodes.append(new_episode)

    # Augment original dataset
    new_df = pd.concat([df] + new_episodes, axis=0).reset_index(drop=True)
    print('Total episodes: %d' % len(new_df.episode.unique()))
    return new_df


def fit_behavior_cloning(experiment,
                         encoder,
                         classifier,
                         train_data,
                         valid_data,
                         timedelta='4h',
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
        print('\nOversampling episodes with infrequent VP doses')
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
    train_dataloader = PrioritizedReplay(train_data, alpha=0.0, timedelta=timedelta, return_history=True, device=device)

    # Callback to evaluate model on valid data
    valid_callback = EvaluationCallback(valid_data, maxlen=truncate)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        total_batches = 0

        # Update learning rate acc. to scheduler
        scheduler.set(optimizer=optimizer, epoch=ep)

        for _ in tqdm(range(batches_per_epoch), desc='Ep %d' % ep):
            # Sample batch of (s, a) pairs from PER
            states, actions, _, _, _, _ = train_dataloader.sample(N=batch_size)

            # Compute error of model on training data
            loss = criterion(model(states), actions.flatten())
            total_loss += loss.item()
            total_batches += 1

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if ep == 0:
                break

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
