import os
import torch
import pickle
import json
import numpy as np
from collections import defaultdict
from glob import glob


class PerformanceTracker:
    def __init__(self, experiment_name, window=5):
        # Store values in here
        self._scores = defaultdict(list)
        self._metrics = []

        # Smooth estimates over window as they can be high variance (for printing!)
        self._window = window

        # Create results directory
        self._path = os.path.join(os.getcwd(), experiment_name)
        num_prev = len(glob(self._path + '_*'))
        self._path = self._path + '_' + str(num_prev).zfill(5)  # e.g. ckcnn_experiment_00001
        os.makedirs(self._path)

    def add(self, **kwargs):
        for metric, value in kwargs.items():
            if metric not in self._metrics:
                self._metrics.append(metric)
            self._scores[metric].append(value)

    def new_best(self, metric, maximize=True):
        """
            Checks whether last score on metric was best seen during training.
            Note: As performance estimates can be high variance (e.g. WIS) we
            average scores over a window (set by `self._smoothing`) to determine improvement.
        """
        scores = self._scores[metric]
        if len(scores) < self._window + 1:
            return False

        # Smooth estimates over window of scores (set by `self._window`)
        scores = [np.mean(scores[i: i + self._window]) for i in range(len(scores) - self._window + 1)]

        # Was last (smoothed) score the best so far?
        return scores[-1] == (np.max(scores) if maximize else np.min(scores))

    def print_stats(self):
        # Returns string-formatted performance stats averaged over a window (set by `self._smoothing`)
        return ', '.join(['%s = %.3f' % (m, np.mean(self._scores[m][-self._window:])) for m in self._metrics])

    def save_metrics(self):
        # Store metrics as .npy files
        for metric, values in self._scores.items():
            np.savetxt(os.path.join(self._path, metric + '.npy'), np.array(values))

    def save_experiment_config(self, **kwargs):
        # Build config file of serializable key:value pairs
        config = defaultdict(dict)
        for kwarg, dct in kwargs.items():
            for key, value in dct.items():
                if type(value) in [bool, int, float, list, tuple, dict, str]:
                    config[kwarg][key] = value

        # Write parameter dictionary to plain-text JSON
        config_path = os.path.join(self._path, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4)

    def save_model_pt(self, model, label):
        # Pickle model
        path = os.path.join(self._path, label + '.pkl')
        with open(path, 'wb') as f:
            pickle.dump(model, f)

        # Also write state_dict to file for good measure
        torch.save(model.state_dict(), self._path + '/state_dict_%s.pt' % label)