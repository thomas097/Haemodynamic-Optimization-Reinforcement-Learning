import pandas as pd
import torch
import numpy as np
from tqdm import tqdm

from sklearn.metrics import f1_score
from dataloader import DataLoader
from performance_tracking import PerformanceTracker
from losses import cross_entropy


def loss_callback(model, dataloader, batch_size, num_actions, device):
    """ Computes CrossEntropy loss of model on data given by dataloader """
    model.eval()

    losses = []
    with torch.no_grad():
        for x, y_true in dataloader.iterate(batch_size=batch_size):
            y_pred = model(x).detach()
            loss = cross_entropy(y_pred, y_true, mask=(y_true >= 0), num_classes=num_actions, device=device)
            losses.append(loss.item())

    model.train()
    return np.mean(losses)


def f1_callback(model, dataloader, batch_size):
    """ Computed macro-averaged F1 score on data given by dataloader """
    model.eval()

    y_pred, y_true = [], []
    with torch.no_grad():
        for x, y in dataloader.iterate(batch_size=batch_size):
            pred_action = torch.argmax(model(x).detach(), dim=2)
            y_pred.append(pred_action.flatten())
            y_true.append(y.flatten())

    model.train()

    y_pred = torch.cat(y_pred, axis=0).cpu().numpy()
    y_true = torch.cat(y_true, axis=0).cpu().numpy()
    mask = (y_true >= 0)
    return f1_score(y_pred[mask], y_true[mask], average='macro')


def oversample_vasopressors(df, rate):
    """ Oversamples episodes with vasopressors to account for action imbalance """
    # Actions with non-zero vasopressors
    vp_actions = set(range(25)) - {0, 5, 10, 15, 20}

    max_episode_id = np.max(df.episode)
    new_episodes = []

    # Identify episodes with vasopressors
    for _, episode in df.groupby('episode', sort=False):
        has_vp = episode.action.isin(vp_actions).any()
        if has_vp:
            # Copy episode and assign new episode id
            for _ in range(int(rate)):
                max_episode_id += 1
                new_episode = episode.copy()
                new_episode.episode = max_episode_id
                new_episodes.append(new_episode)

    # Augment original dataset
    return pd.concat([df] + new_episodes, axis=0).reset_index(drop=True)


def create_classifier(encoder, classif_layers):
    """ Returns encoder appended with a classification head """
    layers = [encoder]
    for i in range(len(classif_layers) - 1):
        layers.append(torch.nn.Linear(classif_layers[i], classif_layers[i + 1]))
        layers.append(torch.nn.LeakyReLU())
    return torch.nn.Sequential(*layers)


# Fits encoder model to action column in training dataset
def fit_behavior_cloning(experiment_name,
                         encoder,
                         classif_layer_shapes,
                         num_actions,
                         train_dataset,
                         valid_dataset,
                         lrate=1e-4,
                         epochs=100,
                         batch_size=16,
                         truncate=1000,
                         oversample_vaso=1.0,
                         eval_after=10,
                         save_on_best=True):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Running on %s' % str(device).upper())

    # Save config file with encoder/experiment params
    tracker = PerformanceTracker(experiment_name)
    tracker.save_experiment_config(encoder=encoder.config, experiment=locals())

    if oversample_vaso > 1.0:
        print('Oversampling episodes with VP by %d' % oversample_vaso)
        train_dataset = oversample_vasopressors(train_dataset, rate=oversample_vaso)

    #####################
    #     Training      #
    #####################

    # Add classification head to encoder
    classif_layer_shapes = tuple(classif_layer_shapes) + (num_actions,)
    model = create_classifier(encoder, classif_layers=classif_layer_shapes).to(device)

    # Load data as DataLoader
    train_dataloader = DataLoader(train_dataset, device=device)
    valid_dataloader = DataLoader(valid_dataset, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    # Training loop
    model.train()
    for ep in range(epochs):
        total_loss = 0.0
        total_batches = 0

        batches_per_episode = int(np.ceil(train_dataloader.size / batch_size))
        with tqdm(desc='Ep %d' % ep, total=batches_per_episode) as pbar:

            for states, actions in train_dataloader.iterate(batch_size=batch_size):
                # Truncate excessively long sequences
                if states.shape[1] > truncate:
                    i = np.random.randint(0, states.shape[1] - truncate)
                    states = states[:, i:i + truncate]
                    actions = actions[:, i:i + truncate]
                    
                # Compute error
                loss = cross_entropy(y_pred=model(states),
                                     y_true=actions,
                                     mask=(actions >= 0),
                                     num_classes=num_actions,
                                     device=device)
                total_loss += loss.item()
                total_batches += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix({'loss': loss.item()})
                pbar.update(1)

                tracker.add(train_loss=loss.item())

                if ep == 0:
                    break

        ############################
        #   Performance Tracking   #
        ############################

        if ep % eval_after == 0:
            # Calc F1 and loss on validation set
            valid_loss = loss_callback(model, valid_dataloader, batch_size, num_actions=num_actions, device=device)
            valid_f1 = f1_callback(model, valid_dataloader, batch_size)

            tracker.add(valid_loss=valid_loss)
            tracker.add(valid_f1=valid_f1)
            tracker.save_metrics()
            print('Ep %d: %s' % (ep, tracker.print_stats()))

            # Save models upon improvement
            new_best = tracker.new_best(metric='valid_loss')
            if save_on_best and new_best:
                print('Model improved! Saving...')
                tracker.save_model_pt(encoder, 'encoder')  # Only need to save the encoder

    model.eval()
    print('Done!')

