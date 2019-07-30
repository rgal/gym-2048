"""Manage training data"""

from __future__ import print_function

import copy
import numpy as np

class training_data(object):

    def __init__(self):
        self._x = np.empty([0, 4, 4], dtype=np.int)
        self._y_digit = np.zeros([0, 1], dtype=np.int)
        self._reward = np.zeros([0, 1], dtype=np.float)
        self._next_x = np.empty([0, 4, 4], dtype=np.int)
        self._done = np.empty([0, 1], dtype=np.bool)

    def copy(self):
        return copy.deepcopy(self)

    def _check_lengths(self):
        assert self._x.shape[0] == self._y_digit.shape[0]
        assert self._x.shape[0] == self._reward.shape[0]
        assert self._x.shape[0] == self._next_x.shape[0]
        assert self._x.shape[0] == self._done.shape[0]

    def get_x(self):
        return self._x

    def get_y_digit(self):
        return self._y_digit

    def get_y_one_hot(self):
        items = self.size()
        one_hot = np.zeros((items, 4))
        flat_y = np.reshape(self._y_digit, (-1, ))
        one_hot[np.arange(items), flat_y] = 1
        return one_hot

    def get_reward(self):
        return self._reward

    def get_next_x(self):
        return self._next_x

    def get_done(self):
        return self._done

    def add(self, board, action, reward, next_board=None, done=False):
        assert reward is not None
        self._x = np.append(self._x, np.reshape(board, (1, 4, 4)), axis=0)

        y_digit = np.zeros([1, 1], dtype=np.int)
        y_digit[0, 0] = action
        self._y_digit = np.append(self._y_digit, y_digit, axis=0)

        r = np.zeros([1, 1], dtype=np.float)
        r[0, 0] = reward
        self._reward = np.append(self._reward, r, axis=0)

        self._next_x = np.append(self._next_x, np.reshape(next_board, (1, 4, 4)), axis=0)

        done_array = np.zeros([1, 1], dtype=np.bool)
        done_array[0, 0] = done
        self._done = np.append(self._done, done_array, axis=0)

        self._check_lengths()

    def get_n(self, n):
        """Get training sample number n"""
        return self._x[n,:,:], self._y_digit[n,:], self._reward[n,:], self._next_x[n,:,:], self._done[n,:]

    def get_total_reward(self):
        """Calculate total reward over all training data, regardless of game ends."""
        return np.sum(self.get_reward())

    def get_highest_tile(self):
        """Get the highest tile on any board. Check the next board as that could be higher."""
        return np.max(self.get_next_x())

    def log2_rewards(self):
        """log2 of reward values to keep them in a small range"""
        items = self.size()
        rewards = np.reshape(self._reward, (items))
        log_rewards = np.ma.log(rewards) / np.ma.log(2)
        self._reward = np.reshape(np.array(log_rewards, np.float), (items, 1))

    def get_lambda_return(self, llambda=0.9):
        """Calculate lambda return from rewards.
           Relies on the data being still in game order. done indicates end
           of episode."""
        items = self.size()
        rewards = list(np.reshape(self._reward, (items)))
        done_list = list(np.reshape(self._done, (items)))
        smoothed_rewards = list()
        previous = None
        rewards.reverse()
        done_list.reverse()
        for i, r in enumerate(rewards):
            smoothed = r
            if done_list[i]:
                previous = None
            if previous:
                smoothed += llambda * previous
            smoothed_rewards.append(smoothed)
            previous = smoothed
        smoothed_rewards.reverse()
        return np.reshape(np.array(smoothed_rewards, np.float), (items, 1))

    def normalize_boards(self, mean=None, sd=None):
        """Normalize boards by subtracting mean and dividing by stdandard deviation"""
        items = self.size()
        boards = self._x
        if mean is None:
            mean = np.mean(boards)
        if sd is None:
            sd = np.std(boards)
        norm_boards = (boards - mean) / sd
        self._x = norm_boards
        norm_next_boards = (self._next_x - mean) / sd
        self._next_x = norm_next_boards

    def normalize_rewards(self, mean=None, sd=None):
        """Normalize rewards by subtracting mean and dividing by stdandard deviation"""
        items = self.size()
        rewards = self._reward
        if mean is None:
            mean = np.mean(rewards)
        if sd is None:
            sd = np.std(rewards)
        norm_rewards = (rewards - mean) / sd
        self._reward = norm_rewards

    def merge(self, other):
        self._x = np.concatenate((self._x, other.get_x()))
        self._y_digit = np.concatenate((self._y_digit, other.get_y_digit()))
        self._reward = np.concatenate((self._reward, other.get_reward()))
        self._next_x = np.concatenate((self._next_x, other.get_next_x()))
        self._done = np.concatenate((self._done, other.get_done()))
        self._check_lengths()

    def split(self, split=0.5):
        splitpoint = int(self.size() * split)
        a = training_data()
        b = training_data()
        a._x = self._x[:splitpoint,:,:]
        b._x = self._x[splitpoint:,:,:]
        a._y_digit = self._y_digit[:splitpoint,:]
        b._y_digit = self._y_digit[splitpoint:,:]
        a._reward = self._reward[:splitpoint,:]
        b._reward = self._reward[splitpoint:,:]
        a._next_x = self._next_x[:splitpoint,:,:]
        b._next_x = self._next_x[splitpoint:,:,:]
        a._done = self._done[:splitpoint,:]
        b._done = self._done[splitpoint:,:]
        a._check_lengths()
        b._check_lengths()
        return a, b

    def sample(self, index_list):
        indexes = np.asarray(index_list)
        sample = training_data()
        sample._x = self._x[indexes,:,:]
        sample._y_digit = self._y_digit[indexes,:]
        sample._reward = self._reward[indexes,:]
        sample._next_x = self._next_x[indexes,:,:]
        sample._done = self._done[indexes,:]
        sample._check_lengths()
        return sample

    def size(self):
        return self._x.shape[0]

    def import_csv(self, filename):
        """Load data as CSV file"""
        # Load board state
        flat_data = np.loadtxt(filename, dtype=np.int, delimiter=',', skiprows=1, usecols=tuple(range(16)))
        self._x = np.reshape(flat_data, (-1, 4, 4))

        # Load actions
        digits = np.loadtxt(filename, dtype=np.int, delimiter=',', skiprows=1, usecols=16)
        self._y_digit = np.reshape(digits, (-1, 1))

        # Load rewards
        reward_data = np.loadtxt(filename, delimiter=',', skiprows=1, usecols=17)
        self._reward = reward_data.reshape(-1, 1)

        # Load next board state
        flat_data = np.loadtxt(filename, dtype=np.int, delimiter=',', skiprows=1, usecols=tuple(range(18, 34)))
        self._next_x = np.reshape(flat_data, (-1, 4, 4))

        # Load dones
        done_data = np.loadtxt(filename, dtype=np.bool, delimiter=',', skiprows=1, usecols=34)
        self._done = done_data.reshape(-1, 1)

        self._check_lengths()

    def construct_header(self, add_returns=False):
        header = list()
        for m in range(1, 5):
            for n in range(1, 5):
                header.append('{}-{}'.format(m, n))
        header.append('action')
        header.append('reward')
        for m in range(1, 5):
            for n in range(1, 5):
                header.append('next {}-{}'.format(m, n))
        header.append('done')
        if add_returns:
            header.append('return')
        return header

    def export_csv(self, filename, add_returns=False):
        """Save data as CSV file"""
        items = self.size()
        flat_x = np.reshape(self._x, (items, 16))
        flat_data = np.concatenate((flat_x, self._y_digit), axis=1)
        flat_data = np.concatenate((flat_data, self._reward), axis=1)
        # Should have flat 16 square board, direction and reward
        assert flat_data.shape[1] == 18
        flat_next_x = np.reshape(self._next_x, (items, 16))
        flat_data = np.concatenate((flat_data, flat_next_x), axis=1)
        assert flat_data.shape[1] == 34

        flat_data = np.concatenate((flat_data, self._done), axis=1)

        if add_returns:
            flat_data = np.concatenate((flat_data, self.get_lambda_return()), axis=1)
        header = self.construct_header(add_returns)

        fformat = '%d,' * 17 + '%f,' + '%d,' * 16 + '%i'
        if add_returns:
            fformat += ',%f'
        np.savetxt(filename, flat_data, comments='', fmt=fformat, header=','.join(header))

    def dump(self):
        print(self._x)
        print(self._y_digit)
        print(self._reward)
        print(self._next_x)
        print(self._done)

    def hflip(self):
        """Flip all the data horizontally"""
        # Add horizontal flip of inputs and outputs
        self._x = np.flip(self.get_x(), 2)

        output_digits = self.get_y_digit()
        # Swap directions 1 and 3
        temp = np.copy(output_digits)
        temp[temp == 1] = 33
        temp[temp == 3] = 1
        temp[temp == 33] = 3
        self._y_digit = temp

        self._next_x = np.flip(self.get_next_x(), 2)

        self._check_lengths()

    def rotate(self, k):
        """Rotate the board by k * 90 degrees"""
        self._x = np.rot90(self.get_x(), k=k, axes=(2, 1))
        self._y_digit = np.mod(self.get_y_digit() + k, 4)
        self._next_x = np.rot90(self.get_next_x(), k=k, axes=(2,1))
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

    def shuffle(self):
        """Shuffle data."""
        self._check_lengths()
        p = np.random.permutation(len(self._x))
        self._x = self._x[p]
        self._y_digit = self._y_digit[p]
        self._reward = self._reward[p]
        self._next_x = self._next_x[p]
        self._done = self._done[p]
