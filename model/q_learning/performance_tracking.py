import os
import torch
import pickle
import numpy as np
from collections import defaultdict
from datetime import datetime


class PerformanceTracker:
    def __init__(self, experiment_name, window=5):
        # Store values in here
        self._scores = defaultdict(list)
        self._metric_lst = []

        # Smooth estimates over window as they can be high variance
        # (only used for printing stats!)
        self._window = window

        # Create folder to store final performance stats into
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        self._full_path = os.path.join(os.getcwd(), experiment_name + '_' + timestamp)
        os.makedirs(self._full_path)

    def add(self, **kwargs):
        for metric, value in kwargs.items():
            if metric not in self._metric_lst:
                self._metric_lst.append(metric)
            self._scores[metric].append(value)

    def new_best(self, metric, maximize=True):
        """ Checks whether improvement in metric was best seen during training.
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
        return ', '.join(['%s = %.3f' % (m, np.mean(self._scores[m][-self._window:])) for m in self._metric_lst])

    def save_metrics(self):
        # Store metrics as .npy
        for metric, values in self._scores.items():
            np.savetxt(os.path.join(self._full_path, metric + '.npy'), np.array(values))

    def save_model_pt(self, model, label):
        # Write model to file using pickle
        pkl_path = os.path.join(self._full_path, label + '.pkl')
        with open(pkl_path, 'wb') as outfile:
            pickle.dump(model, outfile)

        # Write state_dict to file for good measure
        torch.save(model.state_dict(), self._full_path + '/state_dict_%s.pt' % label)