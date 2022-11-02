import os
import copy
import json
import torch
import pickle
import numpy as np
from glob import glob
from collections import defaultdict


class PerformanceTracker:
    def __init__(self, experiment_name):
        # Store values in here
        self._scores = defaultdict(list)
        self._metrics = []

        # Create results directory
        self._path = os.path.join(os.getcwd(), experiment_name)
        num_prev = len(glob(self._path + '_*'))
        self._path = self._path + '_' + str(num_prev).zfill(5)  # e.g. ckcnn_experiment_00001
        os.makedirs(self._path)

    def add(self, **kwargs):
        """ Add metric=value pair to PerformanceTracker """
        for metric, value in kwargs.items():
            if metric not in self._metrics:
                self._metrics.append(metric)
            self._scores[metric].append(value)

    def new_best(self, metric, maximize=True):
        """ Checks whether last score on metric was best seen during training.
        """
        if metric not in self._metrics:
            return False

        func = np.max if maximize else np.min
        return self._scores[metric][-1] == func(self._scores[metric])

    def print_stats(self):
        """ Returns string-formatted performance stats """
        return ', '.join(['%s = %.3f' % (m, self._scores[m][-1]) for m in self._metrics])

    def save_metrics(self):
        """ Store metrics as .npy files """
        for metric, values in self._scores.items():
            np.savetxt(os.path.join(self._path, metric + '.npy'), np.array(values))

    def save_experiment_config(self, **kwargs):
        """ Saves configuration of experiment and model hyperparameters """
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
        """ Pickles model to <label>.pt along with state dict """
        # Pickle model and save state_dict
        torch.save(model, os.path.join(self._path, label + '.pt'))
        torch.save(model.state_dict(), os.path.join(self._path, 'state_dict_%s.pt' % label))