#!/usr/bin/env python

from __future__ import absolute_import
import unittest
import numpy as np
import os
import tempfile

import training_data

class TestTrainingData(unittest.TestCase):
    def test_add(self):
        # Test add with reward
        td = training_data.training_data()
        self.assertTrue(np.array_equal(td.get_x(), np.empty([0, 4, 4], dtype=np.int)))
        self.assertTrue(np.array_equal(td.get_y_digit(), np.empty([0, 1], dtype=np.int)))
        self.assertTrue(np.allclose(td.get_reward(), np.empty([0, 1], dtype=np.float)))
        td.add(np.ones([1, 4, 4]), 1, 4)
        self.assertTrue(np.array_equal(td.get_x(), np.ones([1, 4, 4], dtype=np.int)))
        self.assertTrue(np.array_equal(td.get_y_digit(), np.array([[1]], dtype=np.int)))
        self.assertTrue(np.allclose(td.get_reward(), np.array([[4]], dtype=np.float)))

    def test_get_n(self):
        # Test get_n with reward
        td = training_data.training_data()
        td.add(np.ones([4, 4]), 1, 4)
        td.add(np.zeros([4, 4]), 2, 8)
        (state, action, reward) = td.get_n(1)
        self.assertTrue(np.array_equal(state, np.zeros([4, 4], dtype=np.int)))
        self.assertEqual(action, 2)
        self.assertAlmostEqual(reward, 8)

    def test_hflip(self):
        td = training_data.training_data()
        board1 = np.array([[1, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        td.add(board1, 1, 2)
        board2 = np.array([[0, 0, 0, 0],
                           [2, 4, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        td.add(board2, 2, 0)
        td.hflip()
        expected_x = np.array([
            [[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 4, 2], [0, 0, 0, 0], [0, 0, 0, 0]]
            ], dtype=np.int)
        expected_y_digit = np.array([
            [3],
            [2]
            ], dtype=np.int)
        expected_reward = np.array([
            [2],
            [0],
            ], dtype=np.float)
        self.assertTrue(np.array_equal(td.get_x(), expected_x))
        self.assertTrue(np.array_equal(td.get_y_digit(), expected_y_digit))
        self.assertTrue(np.allclose(td.get_reward(), expected_reward))

    def test_rotate(self):
        td = training_data.training_data()
        board1 = np.array([[1, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        td.add(board1, 1, 2)
        board2 = np.array([[0, 0, 0, 0],
                           [2, 4, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        td.add(board2, 2, 0)
        td.rotate(3)
        expected_x = np.array([
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 4, 0, 0], [0, 2, 0, 0]]
            ], dtype=np.int)
        expected_y_digit = np.array([
            [0],
            [1]
            ], dtype=np.int)
        expected_reward = np.array([
            [2],
            [0],
            ], dtype=np.float)
        self.assertTrue(np.array_equal(td.get_x(), expected_x))
        self.assertTrue(np.array_equal(td.get_y_digit(), expected_y_digit))
        self.assertTrue(np.allclose(td.get_reward(), expected_reward))

    def test_augment(self):
        td = training_data.training_data()
        initial_board = np.array([[1, 1, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]])
        td.add(initial_board, 1, 4)
        td.augment()
        self.assertEqual(td.size(), 8)
        expected_x = np.array([
            [[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 1]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 1]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            [[1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            ], dtype=np.int)
        expected_y_digit = np.array([
            [1],
            [3],
            [2],
            [0],
            [3],
            [1],
            [0],
            [2]
            ], dtype=np.int)
        expected_reward = np.array([
            [4],
            [4],
            [4],
            [4],
            [4],
            [4],
            [4],
            [4]
            ], dtype=np.float)
        self.assertTrue(np.array_equal(td.get_x(), expected_x))
        self.assertTrue(np.array_equal(td.get_y_digit(), expected_y_digit))
        self.assertTrue(np.allclose(td.get_reward(), expected_reward))

    def test_merge(self):
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 1, 16)
        td2 = training_data.training_data()
        td2.add(np.zeros([1, 4, 4]), 2, 0)
        td.merge(td2)
        expected_x = np.array([
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            ], dtype=np.int)
        expected_y_digit = np.array([
            [1],
            [2]
            ], dtype=np.int)
        expected_reward = np.array([
            [16],
            [0]
            ], dtype=np.float)
        self.assertTrue(np.array_equal(td.get_x(), expected_x))
        self.assertTrue(np.array_equal(td.get_y_digit(), expected_y_digit))
        self.assertTrue(np.allclose(td.get_reward(), expected_reward))

    def test_split(self):
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 1, 16)
        td2 = training_data.training_data()
        td2.add(np.zeros([1, 4, 4]), 2, 0)
        td.merge(td2)
        a, b = td.split()
        self.assertTrue(np.array_equal(a.get_x(), np.ones([1, 4, 4])))
        self.assertTrue(np.array_equal(a.get_y_digit(), [[1]]))
        self.assertTrue(np.array_equal(b.get_x(), np.zeros([1, 4, 4])))
        self.assertTrue(np.array_equal(b.get_y_digit(), [[2]]))

    def test_size(self):
        td = training_data.training_data()
        self.assertEqual(td.size(), 0)
        td.add(np.ones([1, 4, 4]), 0, 4)
        self.assertEqual(td.size(), 1)

    def test_log2_rewards(self):
        # Set up training data
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 0, 0)
        td.add(np.ones([1, 4, 4]), 1, 2)
        td.add(np.ones([1, 4, 4]), 2, 4)
        td.add(np.ones([1, 4, 4]), 3, 16)
        td.add(np.ones([1, 4, 4]), 0, 75)
        td.add(np.ones([1, 4, 4]), 1, 2048)
        td.log2_rewards()
        expected_reward = np.array([
            [0], [1], [2], [4], [6.2288], [11]
            ], dtype=np.float)
        self.assertTrue(np.allclose(td.get_reward(), expected_reward))
        expected_action = np.array([
            [0], [1], [2], [3], [0], [1]
            ], dtype=np.int)
        self.assertTrue(np.allclose(td.get_y_digit(), expected_action))

    def test_smooth_rewards(self):
        # Set up training data
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 0, 4)
        td.add(np.ones([1, 4, 4]), 1, 2)
        td.add(np.ones([1, 4, 4]), 2, 16)
        td.add(np.ones([1, 4, 4]), 3, 2)

        # Test using default lambda value of 0.9
        td2 = td.copy()
        td2.smooth_rewards()
        expected_reward = np.array([
            [20.218], [18.02], [17.8], [2.0]
            ], dtype=np.float)
        self.assertTrue(np.allclose(td2.get_reward(), expected_reward))

        # Test using lambda value of 0, should have no effect on rewards
        td2 = td.copy()
        td2.smooth_rewards(llambda=0.0)
        expected_reward = np.array([
            [4], [2], [16], [2]
            ], dtype=np.float)
        self.assertTrue(np.allclose(td2.get_reward(), expected_reward))

    def test_normalize_rewards(self):
        # Test calculating mean and standard deviation
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 1, 4)
        td.add(np.ones([1, 4, 4]), 2, 4)
        td.add(np.ones([1, 4, 4]), 3, 8)
        td.add(np.ones([1, 4, 4]), 0, 16)
        td.normalize_rewards()
        expected_reward = np.array([
            [-0.8165], [-0.8165], [0.], [1.633],
            ], dtype=np.float)
        self.assertTrue(np.allclose(td.get_reward(), expected_reward))
        # Test specifying mean and standard deviation
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 1, 4)
        td.add(np.ones([1, 4, 4]), 2, 4)
        td.add(np.ones([1, 4, 4]), 3, 8)
        td.add(np.ones([1, 4, 4]), 0, 16)
        td.normalize_rewards(mean=8, sd=1)
        expected_reward = np.array([
            [-4.], [-4.], [0.], [8.],
            ], dtype=np.float)
        self.assertTrue(np.allclose(td.get_reward(), expected_reward))

    def test_save_restore(self):
        # Set up training data
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 0, 4)
        td.add(np.zeros([1, 4, 4]), 1, 2)
        td.add(np.ones([1, 4, 4]), 2, 16)
        td.add(np.zeros([1, 4, 4]), 3, 2)

        f = tempfile.NamedTemporaryFile()
        td.export_csv(f.name)

        td2 = training_data.training_data()
        td2.import_csv(f.name)

        expected_x = np.array([
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            ], dtype=np.int)
        expected_y_digit = np.array([
            [0],
            [1],
            [2],
            [3]
            ], dtype=np.int)
        expected_reward = np.array([
            [4],
            [2],
            [16],
            [2]
            ], dtype=np.float)
        self.assertTrue(np.array_equal(td2.get_x(), expected_x))
        self.assertTrue(np.array_equal(td2.get_y_digit(), expected_y_digit))
        self.assertTrue(np.allclose(td2.get_reward(), expected_reward))
        os.remove(f.name)

if __name__ == '__main__':
    unittest.main()
