import os
import torch
import pickle
import numpy as np
from collections import defaultdict
from datetime import datetime


class PerformanceTracker:
    def __init__(self, experiment_name):
        # Store values in here
        self._metrics = defaultdict(list)
        self._names = []

        # Create folder to store final performance stats into
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._full_path = os.path.join(os.getcwd(), experiment_name + '_' + timestamp)
        os.makedirs(self._full_path)

    def add(self, metric_name, value):
        if metric_name not in self._names:
            self._names.append(metric_name)

        if torch.is_tensor(value):
            value = value.item()

        self._metrics[metric_name].append(value)

    def print_stats(self, window=1):
        # Print stats averaged over a window
        return ', '.join(['%s = %.3f' % (m, np.mean(self._metrics[m][-window:])) for m in self._names])

    def save_metrics(self):
        # Store metrics as .npy
        for metric_name, values in self._metrics.items():
            np.savetxt(os.path.join(self._full_path, metric_name + '.npy'), np.array(values))

    def save_model_pt(self, model, model_name):
        # Write model to file using pickle
        pkl_path = os.path.join(self._full_path, model_name + '.pkl')
        with open(pkl_path, 'wb') as outfile:
            pickle.dump(model, outfile)

        # Write state_dict to file
        torch.save(model.state_dict(), self._full_path + '/state_dict_%s.pt' % model_name)