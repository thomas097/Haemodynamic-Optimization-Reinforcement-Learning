import os
import torch
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
        self.save()  # to make sure we won't lose progress on abort!

    def print_stats(self):
        return ', '.join(['%s = %.3f' % (m, self._metrics[m][-1]) for m in self._names])

    def save(self):
        # Store metrics as .npy
        for metric_name, values in self._metrics.items():
            np.savetxt(os.path.join(self._full_path, metric_name + '.npy'), np.array(values))

