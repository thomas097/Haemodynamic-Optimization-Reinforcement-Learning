import torch
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm

from dataloader import DataLoader


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


def fit_behavior_cloning(experiment_name,
                         encoder,
                         classif_layer_sizes,
                         dataset,
                         callback=None,
                         lrate=1e-3,
                         epochs=100,
                         batch_size=32,
                         eval_after=10,
                         save_on_best=True):

    # Define model as encoder with dense classification head
    dense_layers = []
    for i in range(len(classif_layer_sizes) - 1):
        linear = torch.nn.Linear(classif_layer_sizes[i], classif_layer_sizes[i + 1])
        dense_layers.extend([linear, torch.nn.LeakyReLU()])
    model = torch.nn.Sequential(encoder, *dense_layers)

    criterion = CrossEntropy(num_classes=classif_layer_sizes[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lrate)

    #####################
    #     Training      #
    #####################

    # Place model and training data on same device (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print('Running on %s' % device)

    # Place dataset into randomized replay buffer
    dataloader = DataLoader(dataset, device=device)

    # Training loop
    for ep in range(epochs):
        total_loss = 0.0
        total_batches = 0

        for x, y_true in tqdm(dataloader.iterate(batch_size=batch_size), desc='Ep %d' % ep):
            # Safety fallback: skip extremely long sequences!
            if x.shape[1] > 1000:
                continue

            loss = criterion(model(x), y_true)
            total_loss += loss.item()
            total_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if total_batches > 100:
                break

        if ep % eval_after == 0:
            print('Ep %d: CE_loss = %.3f' % (ep, total_loss / total_batches))
