import torch
import random
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from experience_replay import EvaluationReplay
from attention_models import CausalTransformer
from utils import load_data, count_parameters


class Callback:
    """ Callback to run model on a (shortened) validation set """
    def __init__(self, valid_df, batch_size=128):
        self._dataloader = EvaluationReplay(valid_df, return_history=True)
        self._labels = labels_from_rewards(valid_df)
        self._batch_size = batch_size
        self._batches_per_episode = int(np.ceil(len(self._labels) / batch_size))

    def __call__(self, model):
        y_pred = []
        with torch.no_grad():
            for x in self._dataloader.iterate(self._batch_size):
                y_pred.append(np.argmax(model(x).detach().numpy(), axis=1))
        pred_labels = np.concatenate(y_pred, axis=0)

        return accuracy_score(self._labels, pred_labels)


def labels_from_rewards(df):
    """ Converts the end-of-episode reward labels to mortality labels (0=deceased; 1=survived)
    """
    def to_mortality(rewards):
        final_reward = np.max(rewards)
        return np.ones(len(rewards)) if final_reward == 15 else np.zeros(len(rewards))

    non_nan_states = df[df.reward.notna()]
    return non_nan_states.groupby('episode', sort=False).reward.transform(to_mortality).values


def downsample(df):
    # Down-sample dataset so we have same the number of 0/1 labels
    outcomes = df.groupby('episode').reward.max()
    pos_rewards = outcomes == 15
    neg_rewards = outcomes != 15

    min_size = min(np.sum(pos_rewards), np.sum(neg_rewards))

    pos_episodes = list(outcomes.index[pos_rewards])
    neg_episodes = list(outcomes.index[neg_rewards])
    random.shuffle(pos_episodes)
    random.shuffle(neg_episodes)

    # Select equal number of random episodes with each outcome
    filtered_episodes = pos_episodes[:min_size] + neg_episodes[:min_size]
    out_df = df[df.episode.isin(filtered_episodes)]

    print('Number of episodes:', len(out_df.episode.unique()))
    return out_df


def fit_encoder_mortality_prediction(train_df, callback, encoder, encoder_out_channels, lrate=1e-3, epochs=10,
                                     batch_size=32, eval_every=200):
    """ Pretrains an encoder model using semi-supervised pretraining on a mortality prediction task
    """
    # Load training data into dataloader
    train_dataloader = EvaluationReplay(train_df,  return_history=True)

    # Append classification head to encoder
    layer1 = torch.nn.Linear(encoder_out_channels, 32)
    relu = torch.nn.LeakyReLU()
    layer2 = torch.nn.Linear(32, 2)
    model = torch.nn.Sequential(encoder, layer1, relu, layer2)

    # Infer 0/1 label from [-15, 15] rewards at end-of-episode
    labels = torch.LongTensor(labels_from_rewards(train_df))

    num_deceased = torch.sum(labels == 0)
    num_survived = torch.sum(labels == 1)
    print('\ndeceased:', num_deceased)
    print('survived:', num_survived)

    # Loss and optimizer (weighted by class support to balance out loss!)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    num_batches_per_episode = int(np.ceil(len(labels) / batch_size))
    if eval_every > num_batches_per_episode:
        raise Exception('eval_every > num_batches_per_episode')

    model.train()
    for ep in range(epochs):
        total_loss = []

        for i, batch in tqdm(enumerate(train_dataloader.iterate(batch_size)), total=num_batches_per_episode):
            y_pred = model(batch)
            y_true = labels[i * batch_size: (i + 1) * batch_size]

            loss = criterion(y_pred, y_true)
            total_loss.append(loss.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Evaluate every few batches
            if i % eval_every == 0:
                model.eval()
                valid_f1 = callback(model)
                model.train()

                print('\nEp %d: CE = %.3f, F1 = %.3f' % (ep, np.mean(total_loss), valid_f1))


if __name__ == '__main__':
    train_df = downsample(load_data('../preprocessing/datasets/mimic-iii/attention_4h_with_cv/mimic-iii_train.csv'))
    valid_df = downsample(load_data('../preprocessing/datasets/mimic-iii/attention_4h_with_cv/mimic-iii_valid.csv'))

    encoder = CausalTransformer(vocab_size=45, d_model=24, out_channels=64, nheads=1, truncate=256)
    print('Encoder params:', count_parameters(encoder))

    # Callback to measure validation performance
    callback = Callback(valid_df)

    fit_encoder_mortality_prediction(train_df, callback, encoder, encoder_out_channels=64)
