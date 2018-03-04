"""Manage training data"""

from __future__ import print_function

import numpy as np
import os

class training_data(object):

    def __init__(self):
        self._x = np.empty([0, 4, 4], dtype=np.int)
        self._y = np.zeros([0, 4], dtype=np.int)

    def _check_lengths(self):
        assert self._x.shape[0] == self._y.shape[0]

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def add(self, board, action):
        self._x = np.append(self._x, np.reshape(board, (1, 4, 4)), axis=0)
        y = np.zeros([1, 4], dtype=np.int)
        y[0, action] = 1
        self._y = np.append(self._y, y, axis=0)

    def merge(self, other):
        self._x = np.concatenate((self._x, other.get_x()))
        self._y = np.concatenate((self._y, other.get_y()))
        self._check_lengths()

    def get_x_partitioned(self):
        x_copy = np.copy(self._x)
        np.random.shuffle(x_copy)
        split_point = int(self.size() * 0.9)
        x_train = x_copy[:split_point,:,:]
        x_dev = x_copy[split_point:,:,:]
        return (x_train, x_dev)

    def get_y_partitioned(self):
        y_copy = np.copy(self._y)
        np.random.shuffle(y_copy)
        split_point = int(self.size() * 0.9)
        y_train = y_copy[:split_point,:]
        y_dev = y_copy[split_point:,:]
        return (y_train, y_dev)

    def size(self):
        return self._x.shape[0]

    def read(self, input_dir):
        with open(os.path.join(input_dir, 'data.npy'), 'r') as f:
            data = np.load(f)
            self._x = data['x']
            self._y = data['y']
        self._check_lengths()

    def write(self, output_dir):
        # Save training data
        try:
            os.makedirs(output_dir)
        except OSError:
            pass
        with open(os.path.join(output_dir, 'data.npy'), 'w') as f:
            print("Outputting {} with shapes {} {}".format('data.npy', self._x.shape, self._y.shape))
            np.savez(f, x=self._x, y=self._y)

    def export_csv(self, filename):
        """Save data as CSV file"""
        items = self.size()
        flat_x = np.reshape(self._x, (items, 16))
        flat_data = np.concatenate((flat_x, self._y), axis=1)
        # Should have flat 16 square board and one hot encoded direction
        assert flat_data.shape[1] == 20
        np.savetxt(filename, flat_data, fmt='%d', delimiter=',')

    def dump(self):
        print(self._x)
        print(self._y)

    def augment(self):
        """Flip the board horizontally, then add rotations to other orientations."""
        inputs = self.get_x()
        outputs = self.get_y()
        # Add horizontal flip of inputs and outputs
        flipped_inputs = np.concatenate((inputs, np.flip(inputs, 2)))

        # Swap directions 1 and 3
        temp = np.copy(outputs)
        temp[:,[1,3]] = temp[:,[3,1]]
        flipped_outputs = np.concatenate((outputs, temp))

        # Add 3 rotations of the previous
        augmented_inputs = np.concatenate((flipped_inputs,
            np.rot90(flipped_inputs, k=1, axes=(2, 1)),
            np.rot90(flipped_inputs, k=2, axes=(2, 1)),
            np.rot90(flipped_inputs, k=3, axes=(2, 1))))
        augmented_outputs = np.concatenate((flipped_outputs,
            np.roll(flipped_outputs, 1, axis=1),
            np.roll(flipped_outputs, 2, axis=1),
            np.roll(flipped_outputs, 3, axis=1)))
        self._x = augmented_inputs
        self._y = augmented_outputs
        self._check_lengths()
