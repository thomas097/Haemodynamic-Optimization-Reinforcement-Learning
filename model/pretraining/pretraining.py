import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import DataLoader
from performance_tracking import PerformanceTracker


class CrossEntropy:
    def __init__(self, num_classes):
        self._softmax = torch.nn.Softmax(dim=2)
        self._num_classes = num_classes

    def _one_hot(self, y_true, mask):
        # One-hot encode target (take care of padding/NaNs encoded with -1)
        y_true.masked_fill_(mask, 0)
        y_true = F.one_hot(y_true, self._num_classes).float()
        return y_true, mask

    def __call__(self, y_pred, y_true):
        pred = self._softmax(y_pred)
        target, mask = self._one_hot(y_true, mask=y_true < 0)
        loss = -torch.sum(target * torch.log(pred), dim=2)
        loss.masked_fill_(mask, 0)  # We mask out from loss entries which were once NaNs or padding

        # Scale by to compensate for NaNs
        return torch.mean(loss) * (torch.numel(loss) / torch.sum(~mask))


def eval_callback(model, valid_dataloader, criterion, batch_size):
    """ Returns mean loss of model on validation data given criterion (e.g. CE)
    """
    loss = []
    model.eval()
    with torch.no_grad():
        for x, y in tqdm(valid_dataloader.iterate(batch_size=batch_size), desc='Running valid'):
            loss.append(criterion(model(x), y).item())
    model.train()
    return {'valid_loss': np.mean(loss)}


def fit_behavior_cloning(experiment_name,
                         encoder,
                         classif_layer_sizes,
                         train_dataset,
                         valid_dataset,
                         lrate=1e-4,
                         epochs=100,
                         batch_size=32,
                         truncate=1500,
                         eval_after=10,
                         lambda_reg=1e-5,
                         save_on_best=True):

    # Track performance and hyperparameters of model
    tracker = PerformanceTracker(experiment_name)
    tracker.save_experiment_config(encoder=encoder.config, experiment=locals())

    # Define model as encoder with dense classification head
    dense_layers = []
    for i in range(len(classif_layer_sizes) - 1):
        linear = torch.nn.Linear(classif_layer_sizes[i], classif_layer_sizes[i + 1])
        dense_layers.extend([linear, torch.nn.LeakyReLU()])
    model = torch.nn.Sequential(encoder, *dense_layers)

    #####################
    #     Training      #
    #####################

    # Place model and training data on same device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Running on %s' % device)

    # Load train/val datasets into DataLoaders
    train_dataloader = DataLoader(train_dataset, device=device)
    valid_dataloader = DataLoader(valid_dataset, device=device)
    batches_per_episode = int(np.ceil(train_dataloader.size / batch_size))

    criterion = CrossEntropy(num_classes=classif_layer_sizes[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate, weight_decay=lambda_reg)

    # Training loop
    for ep in range(epochs):
        total_loss = 0.0
        total_batches = 0

        with tqdm(desc='Ep %d' % ep, total=batches_per_episode) as pbar:

            for x, y in train_dataloader.iterate(batch_size=batch_size):
                # Safety fallback: truncate extremely long sequences!
                if x.shape[1] > truncate:
                    x = x[:, -truncate:]
                    y = y[:, -truncate:]

                loss = criterion(model(x), y)
                total_loss += loss.item()
                total_batches += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if ep == 0:
                    break

                pbar.set_postfix({'CE_loss': loss.item()})
                pbar.update(1)

                tracker.add(train_loss=loss.item())

        ############################
        #   Performance Tracking   #
        ############################

        if ep % eval_after == 0:
            tracker.add(**eval_callback(model, valid_dataloader, criterion, batch_size))
            tracker.save_metrics()
            print('Ep %d: %s' % (ep, tracker.print_stats()))

            # Save models upon improvement (or always if save_on=False)
            new_best = tracker.new_best(metric='valid_loss')
            if new_best:
                print('Model improved! Saving...')
                tracker.save_model_pt(encoder, 'encoder')  # Only need to save the encoder
    print('Done!')

