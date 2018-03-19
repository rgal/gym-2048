"""Manage training data"""

from __future__ import print_function

import copy
import numpy as np
import os

class training_data(object):

    def __init__(self):
        self._x = np.empty([0, 4, 4], dtype=np.int)
        self._y = np.zeros([0, 4], dtype=np.int)
        self._y_digit = np.zeros([0, 1], dtype=np.int)

    def copy(self):
        return copy.deepcopy(self)

    def _check_lengths(self):
        assert self._x.shape[0] == self._y.shape[0]
        assert self._y.shape[0] == self._y_digit.shape[0]

    def get_x(self):
        return self._x

    def get_y(self):
        return self._y

    def get_y_digit(self):
        return self._y_digit

    def add(self, board, action):
        self._x = np.append(self._x, np.reshape(board, (1, 4, 4)), axis=0)
        y = np.zeros([1, 4], dtype=np.int)
        y[0, action] = 1
        self._y = np.append(self._y, y, axis=0)

        y_digit = np.zeros([1, 1], dtype=np.int)
        y_digit[0, 0] = action
        self._y_digit = np.append(self._y_digit, y_digit, axis=0)

    def merge(self, other):
        self._x = np.concatenate((self._x, other.get_x()))
        self._y = np.concatenate((self._y, other.get_y()))
        self._y_digit = np.concatenate((self._y_digit, other.get_y_digit()))
        self._check_lengths()

    def split(self, split=0.5):
        splitpoint = int(self.size() * split)
        a = training_data()
        b = training_data()
        a._x = self._x[:splitpoint,:,:]
        b._x = self._x[splitpoint:,:,:]
        a._y = self._y[:splitpoint,:]
        b._y = self._y[splitpoint:,:]
        a._y_digit = self._y_digit[:splitpoint,:]
        b._y_digit = self._y_digit[splitpoint:,:]
        a._check_lengths()
        b._check_lengths()
        return a, b

    def size(self):
        return self._x.shape[0]

    def read(self, input_dir):
        with open(os.path.join(input_dir, 'data.npy'), 'r') as f:
            data = np.load(f)
            self._x = data['x']
            self._y = data['y']
            self._y_digit = np.argmax(self._y, axis=1)
        l = self.size()
        self._y_digit = np.reshape(self._y_digit, (l, 1))
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

    def import_csv(self, filename):
        """Load data as CSV file"""
        flat_data = np.loadtxt(filename, delimiter=',')
        assert flat_data.shape[1] == 17
        items = flat_data.shape[0]
        self._x = np.reshape(flat_data[:,:16], (items, 4, 4))
        y_digits = flat_data[:,16].astype(int)
        # Reconstruct one hot _y
        y = np.zeros([items, 4], dtype=np.int)
        y[np.arange(items), y_digits] = 1
        self._y = y
        self._y_digit = np.reshape(y_digits, (items, 1))
        self._check_lengths()

    def export_csv(self, filename):
        """Save data as CSV file"""
        items = self.size()
        flat_x = np.reshape(self._x, (items, 16))
        flat_data = np.concatenate((flat_x, self._y_digit), axis=1)
        # Should have flat 16 square board and one hot encoded direction
        assert flat_data.shape[1] == 17
        np.savetxt(filename, flat_data, fmt='%d', delimiter=',')

    def dump(self):
        print(self._x)
        print(self._y)
        print(self._y_digit)

    def randomise(self):
        """Randomise orientation of training data"""
        inputs = self.get_x()
        outputs = self.get_y()
        output_digits = self.get_y_digit()
        items = self.size()

    def hflip(self):
        """Flip all the data horizontally"""
        inputs = self.get_x()
        outputs = self.get_y()
        output_digits = self.get_y_digit()

        # Add horizontal flip of inputs and outputs
        self._x = np.flip(inputs, 2)

        # Swap directions 1 and 3
        temp = np.copy(outputs)
        temp[:,[1,3]] = temp[:,[3,1]]
        self._y = temp

        # Swap directions 1 and 3
        temp = np.copy(output_digits)
        temp[temp == 1] = 33
        temp[temp == 3] = 1
        temp[temp == 33] = 3
        self._y_digit = temp
        self._check_lengths()

    def rotate(self, k):
        """Rotate the board by k * 90 degrees"""
        self._x = np.rot90(self.get_x(), k=k, axes=(2, 1))
        self._y = np.roll(self.get_y(), k, axis=1)
        self._y_digit = np.mod(self.get_y_digit() + k, 4)
        self._check_lengths()

    def augment(self):
        """Flip the board horizontally, then add rotations to other orientations."""
        # Add a horizontal flip of the board
        other = self.copy()
        other.hflip()
        self.merge(other)

        # Add 3 rotations of the previous
        k1 = self.copy()
        k2 = self.copy()
        k3 = self.copy()
        k1.rotate(1)
        k2.rotate(2)
        k3.rotate(3)
        self.merge(k1)
        self.merge(k2)
        self.merge(k3)

        self._check_lengths()
