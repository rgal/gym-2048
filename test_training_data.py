#!/usr/bin/env python

from __future__ import absolute_import
import unittest
import numpy as np
import os
import tempfile

import training_data

class TestTrainingData(unittest.TestCase):
    def test_add(self):
        td = training_data.training_data()
        self.assertTrue(np.array_equal(td.get_x(), np.empty([0, 4, 4], dtype=np.int)))
        self.assertTrue(np.array_equal(td.get_y_digit(), np.empty([0, 1], dtype=np.int)))
        self.assertTrue(np.allclose(td.get_reward(), np.empty([0, 1], dtype=np.float)))
        self.assertTrue(np.array_equal(td.get_next_x(), np.empty([0, 4, 4], dtype=np.int)))
        self.assertTrue(np.array_equal(td.get_done(), np.empty([0, 1], dtype=np.bool)))
        td.add(np.ones([1, 4, 4]), 1, 4, np.zeros([1, 4, 4]), True)
        self.assertTrue(np.array_equal(td.get_x(), np.ones([1, 4, 4], dtype=np.int)))
        self.assertTrue(np.array_equal(td.get_y_digit(), np.array([[1]], dtype=np.int)))
        self.assertTrue(np.allclose(td.get_reward(), np.array([[4]], dtype=np.float)))
        self.assertTrue(np.array_equal(td.get_next_x(), np.zeros([1, 4, 4], dtype=np.int)))
        self.assertTrue(np.array_equal(td.get_done(), np.array([[1]], dtype=np.bool)))

    def test_get_n(self):
        td = training_data.training_data()
        td.add(np.ones([4, 4]), 1, 4, np.zeros([4, 4]))
        td.add(np.zeros([4, 4]), 2, 8, np.ones([4, 4]))
        (state, action, reward, next_state, done) = td.get_n(1)
        self.assertTrue(np.array_equal(state, np.zeros([4, 4], dtype=np.int)))
        self.assertEqual(action, 2)
        self.assertAlmostEqual(reward, 8)
        self.assertTrue(np.array_equal(next_state, np.ones([4, 4], dtype=np.int)))

    def test_hflip(self):
        td = training_data.training_data()
        board1 = np.array([[1, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        board2 = np.array([[0, 0, 0, 0],
                           [2, 4, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        td.add(board1, 1, 2, board2)
        td.add(board2, 2, 0, board1)
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
        expected_next_x = np.array([
            [[0, 0, 0, 0], [0, 0, 4, 2], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            ], dtype=np.int)
        self.assertTrue(np.array_equal(td.get_x(), expected_x))
        self.assertTrue(np.array_equal(td.get_y_digit(), expected_y_digit))
        self.assertTrue(np.allclose(td.get_reward(), expected_reward))
        self.assertTrue(np.allclose(td.get_next_x(), expected_next_x))

    def test_rotate(self):
        td = training_data.training_data()
        board1 = np.array([[1, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        board2 = np.array([[0, 0, 0, 0],
                           [2, 4, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        td.add(board1, 1, 2, board2)
        td.add(board2, 2, 0, board1)
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
        expected_next_x = np.array([
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 4, 0, 0], [0, 2, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]
            ], dtype=np.int)
        self.assertTrue(np.array_equal(td.get_x(), expected_x))
        self.assertTrue(np.array_equal(td.get_y_digit(), expected_y_digit))
        self.assertTrue(np.allclose(td.get_reward(), expected_reward))
        self.assertTrue(np.array_equal(td.get_next_x(), expected_next_x))

    def test_augment(self):
        td = training_data.training_data()
        initial_board = np.array([[1, 1, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]])
        next_board = np.array([[0, 0, 0, 2],
                               [0, 2, 0, 0],
                               [0, 0, 0, 0],
                               [0, 0, 0, 0]])
        td.add(initial_board, 1, 4, next_board)
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
        expected_next_x = np.array([
            [[0, 0, 0, 2], [0, 2, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]], # Original
            [[2, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 0]], # Hflip'd
            [[0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0], [0, 0, 0, 2]], # Original, rotated 90 degrees
            [[0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 2, 0], [0, 0, 0, 0]], # Hflip, rotated 90 degrees
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 2, 0], [2, 0, 0, 0]], # Original, rotated 180 degrees
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 2]], # Hflip, rotated 180 degrees
            [[2, 0, 0, 0], [0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0]], # Original, rotate 270 degrees
            [[0, 0, 0, 0], [0, 2, 0, 0], [0, 0, 0, 0], [2, 0, 0, 0]]  # Hflip, rotated 270 degrees
            ], dtype=np.int)
        self.assertTrue(np.array_equal(td.get_x(), expected_x))
        self.assertTrue(np.array_equal(td.get_y_digit(), expected_y_digit))
        self.assertTrue(np.allclose(td.get_reward(), expected_reward))
        self.assertTrue(np.array_equal(td.get_next_x(), expected_next_x))

    def test_merge(self):
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 1, 16, np.zeros([1, 4, 4]))
        td2 = training_data.training_data()
        td2.add(np.zeros([1, 4, 4]), 2, 0, np.ones([1, 4, 4]))
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
        expected_next_x = np.array([
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
            ], dtype=np.int)
        self.assertTrue(np.array_equal(td.get_x(), expected_x))
        self.assertTrue(np.array_equal(td.get_y_digit(), expected_y_digit))
        self.assertTrue(np.allclose(td.get_reward(), expected_reward))
        self.assertTrue(np.array_equal(td.get_next_x(), expected_next_x))

    def test_split(self):
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 1, 16, np.zeros([1, 4, 4]))
        td2 = training_data.training_data()
        td2.add(np.zeros([1, 4, 4]), 2, 0, np.ones([1, 4, 4]))
        td.merge(td2)
        a, b = td.split()
        self.assertTrue(np.array_equal(a.get_x(), np.ones([1, 4, 4])))
        self.assertTrue(np.array_equal(a.get_y_digit(), [[1]]))
        self.assertTrue(np.array_equal(a.get_reward(), [[16]]))
        self.assertTrue(np.array_equal(a.get_next_x(), np.zeros([1, 4, 4])))
        self.assertTrue(np.array_equal(b.get_x(), np.zeros([1, 4, 4])))
        self.assertTrue(np.array_equal(b.get_y_digit(), [[2]]))
        self.assertTrue(np.array_equal(b.get_reward(), [[0]]))
        self.assertTrue(np.array_equal(b.get_next_x(), np.ones([1, 4, 4])))

    def test_sample(self):
        td = training_data.training_data()
        td.add(np.zeros([1, 4, 4]), 0, 0, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 1, 1, np.ones([1, 4, 4]))
        sample = td.sample([1])
        self.assertEqual(sample.size(), 1)
        self.assertIn(sample.get_y_digit(), [[[0]], [[1]]])
        if sample.get_y_digit() == 0:
            self.assertTrue(np.array_equal(sample.get_x(), np.zeros([1, 4, 4])))
        if sample.get_y_digit() == 1:
            self.assertTrue(np.array_equal(sample.get_x(), np.ones([1, 4, 4])))

    def test_size(self):
        td = training_data.training_data()
        self.assertEqual(td.size(), 0)
        td.add(np.ones([1, 4, 4]), 0, 4, np.zeros([1, 4, 4]))
        self.assertEqual(td.size(), 1)

    def test_log2_rewards(self):
        # Set up training data
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 0, 0, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 1, 2, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 2, 4, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 3, 16, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 0, 75, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 1, 2048, np.zeros([1, 4, 4]))
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
        td.add(np.ones([1, 4, 4]), 0, 4, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 1, 2, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 2, 16, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 3, 2, np.zeros([1, 4, 4]))

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

        # Test end of episode
        td3 = training_data.training_data()
        td3.add(np.ones([1, 4, 4]), 0, 4, np.zeros([1, 4, 4]), False)
        td3.add(np.ones([1, 4, 4]), 1, 2, np.zeros([1, 4, 4]), True)
        td3.add(np.ones([1, 4, 4]), 2, 16, np.zeros([1, 4, 4]), False)
        td3.add(np.ones([1, 4, 4]), 3, 2, np.zeros([1, 4, 4]), True)
        td3.smooth_rewards()
        expected_reward = np.array([
            [5.8], [2.0], [17.8], [2.0]
            ], dtype=np.float)
        self.assertTrue(np.allclose(td3.get_reward(), expected_reward))

    def test_normalize_rewards(self):
        # Test calculating mean and standard deviation
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 1, 4, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 2, 4, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 3, 8, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 0, 16, np.zeros([1, 4, 4]))
        td.normalize_rewards()
        expected_reward = np.array([
            [-0.8165], [-0.8165], [0.], [1.633],
            ], dtype=np.float)
        self.assertTrue(np.allclose(td.get_reward(), expected_reward))
        # Test specifying mean and standard deviation
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 1, 4, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 2, 4, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 3, 8, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 0, 16, np.zeros([1, 4, 4]))
        td.normalize_rewards(mean=8, sd=1)
        expected_reward = np.array([
            [-4.], [-4.], [0.], [8.],
            ], dtype=np.float)
        self.assertTrue(np.allclose(td.get_reward(), expected_reward))

    def test_save_restore(self):
        # Set up training data
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 0, 4, np.zeros([1, 4, 4]))
        td.add(np.zeros([1, 4, 4]), 1, 2, np.ones([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 2, 16, np.zeros([1, 4, 4]))
        td.add(np.zeros([1, 4, 4]), 3, 2, np.ones([1, 4, 4]))

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
        expected_next_x = np.array([
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]]
            ], dtype=np.int)
        self.assertTrue(np.array_equal(td2.get_x(), expected_x))
        self.assertTrue(np.array_equal(td2.get_y_digit(), expected_y_digit))
        self.assertTrue(np.allclose(td2.get_reward(), expected_reward))
        self.assertTrue(np.array_equal(td2.get_next_x(), expected_next_x))
        os.remove(f.name)

if __name__ == '__main__':
    unittest.main()
