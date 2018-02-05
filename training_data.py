"""Manage training data"""

from __future__ import print_function

import numpy as np
import os

class training_data(object):

    def __init__(self):
        self._x = np.empty([0, 4, 4], dtype=np.int)
        self._y = np.zeros([0, 4], dtype=np.int)

    def add(self, board, action):
        self._x = np.append(self._x, np.reshape(board, (1, 4, 4)), axis=0)
        y = np.zeros([1, 4], dtype=np.int)
        y[0, action] = 1
        self._y = np.append(self._y, y, axis=0)

    def size(self):
        return self._x.shape[0]

    def write(self, output_dir):
        # Save training data
        try:
            os.makedirs(output_dir)
        except OSError:
            pass
        with open(os.path.join(output_dir, 'x.npy'), 'w') as f:
            print("Outputting {} with shape {}".format('x.npy', self._x.shape))
            np.save(f, self._x)
        with open(os.path.join(output_dir, 'y.npy'), 'w') as f:
            print("Outputting {} with shape {}".format('y.npy', self._y.shape))
            np.save(f, self._y)

    def dump(self):
        print(self._x)
        print(self._y)
