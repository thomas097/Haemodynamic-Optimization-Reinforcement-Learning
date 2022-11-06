import os
import torch
import numpy as np

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
    def _cross_entropy(y_pred, y_true):
        criterion = torch.nn.CrossEntropyLoss()
        return criterion(y_pred, y_true).item()

    @staticmethod
    def _f1_score(y_pred, y_true):
        y_pred = torch.argmax(y_pred, dim=1).detach().numpy()
        y_true = y_true.detach().numpy()
        return f1_score(y_true, y_pred, average='macro')

    def __call__(self, model, batch_size):
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
        scores = {'valid_loss': self._cross_entropy(y_pred, y_true),
                  'valid_f1': self._f1_score(y_pred, y_true)}
        return scores


def get_class_weights(df):
    """ Weigh action classes inversely proportional to their frequency """
    counts = df.action.value_counts()
    weights = torch.zeros(25)
    for i, count in counts.to_dict().items():
        weights[int(i)] = (1 / count) * (counts.sum() / 25)

    print('Class/action weights:')
    print(weights, '\n')
    return weights.detach()


def fit_behavior_cloning(experiment,
                         encoder,
                         classifier,
                         train_data,
                         valid_data,
                         timedelta='4h',
                         batches_per_epoch=100,
                         epochs=100,
                         lrate=1e-3,
                         batch_size=32,
                         truncate=256,
                         save_on=False):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on %s' % device)

    tracker = PerformanceTracker(experiment)
    tracker.save_experiment_config(encoder=encoder.config, experiment=locals())

    # Optimize combined encoder-classifier model w.r.t weighted CE loss
    model = torch.nn.Sequential(encoder, classifier).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)
    criterion = torch.nn.CrossEntropyLoss(weight=get_class_weights(train_data))

    # Gradient clipping to minimize influence of bad batches
    for w in model.parameters():
        if w.requires_grad:
            w.register_hook(lambda grad: torch.clamp(grad, -1, 1))

    #####################
    #     Training      #
    #####################

    # Load training data into replay buffer (set to random sampling)
    train_dataloader = PrioritizedReplay(train_data, alpha=0.0, timedelta=timedelta, return_history=True, device=device)

    # Callback to evaluate model on valid set
    valid_callback = EvaluationCallback(valid_data, maxlen=truncate)

    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        total_batches = 0

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
