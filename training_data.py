"""Manage training data"""

from __future__ import print_function

import numpy as np
import os

class training_data(object):

    def __init__(self):
        self._data = list()

    def add(self, board, action):
        self._data.append({'input': board, 'output': action})

    def size(self):
        return len(self._data)

    def write(self, output_dir):
        x = np.empty([self.size(), 4, 4], dtype=np.int)
        for idx, d in enumerate(self._data):
            x[idx] = np.reshape(d['input'], (4, 4))
        y = np.zeros([self.size(), 4], dtype=np.int)
        for idx, d in enumerate(self._data):
            y[idx, d['output']] = 1


        # Save training data
        try:
            os.makedirs(output_dir)
        except OSError:
            pass
        with open(os.path.join(output_dir, 'x.npy'), 'w') as f:
            np.save(f, x)
        with open(os.path.join(output_dir, 'y.npy'), 'w') as f:
            np.save(f, y)

    def dump(self):
        print(self._data)
