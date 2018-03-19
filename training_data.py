"""Manage training data"""

from __future__ import print_function

import numpy as np
import os

class training_data(object):

    def __init__(self):
        self._x = np.empty([0, 4, 4], dtype=np.int)
        self._y = np.zeros([0, 4], dtype=np.int)
        self._y_digit = np.zeros([0, 1], dtype=np.int)

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
        inputs = self.get_x()
        outputs = self.get_y()
        output_digits = self.get_y_digit()

        # Add horizontal flip of inputs and outputs
        flipped_inputs = np.concatenate((inputs, np.flip(inputs, 2)))

        # Swap directions 1 and 3
        temp = np.copy(outputs)
        temp[:,[1,3]] = temp[:,[3,1]]
        flipped_outputs = np.concatenate((outputs, temp))

        # Swap directions 1 and 3
        temp = np.copy(output_digits)
        temp[temp == 1] = 33
        temp[temp == 3] = 1
        temp[temp == 33] = 3
        flipped_output_digits = np.concatenate((output_digits, temp))

        # Add 3 rotations of the previous
        augmented_inputs = np.concatenate((flipped_inputs,
            np.rot90(flipped_inputs, k=1, axes=(2, 1)),
            np.rot90(flipped_inputs, k=2, axes=(2, 1)),
            np.rot90(flipped_inputs, k=3, axes=(2, 1))))
        augmented_outputs = np.concatenate((flipped_outputs,
            np.roll(flipped_outputs, 1, axis=1),
            np.roll(flipped_outputs, 2, axis=1),
            np.roll(flipped_outputs, 3, axis=1)))
        augmented_output_digits = np.concatenate((flipped_output_digits,
            np.mod(flipped_output_digits + 1, 4),
            np.mod(flipped_output_digits + 2, 4),
            np.mod(flipped_output_digits + 3, 4)
        ))
        self._x = augmented_inputs
        self._y = augmented_outputs
        self._y_digit = augmented_output_digits

        self._check_lengths()
