#!/usr/bin/env python

from __future__ import absolute_import
import unittest
import numpy as np

import training_data

class TestTrainingData(unittest.TestCase):
    def test_add(self):
        td = training_data.training_data()
        self.assertTrue(np.array_equal(td.get_x(), np.empty([0, 4, 4], dtype=np.int)))
        self.assertTrue(np.array_equal(td.get_y(), np.empty([0, 4], dtype=np.int)))
        td.add(np.ones([1, 4, 4]), 1)
        self.assertTrue(np.array_equal(td.get_x(), np.ones([1, 4, 4], dtype=np.int)))
        self.assertTrue(np.array_equal(td.get_y(), np.array([[0, 1, 0, 0]], dtype=np.int)))

    def test_hflip(self):
        td = training_data.training_data()
        board1 = np.array([[1, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        td.add(board1, 1)
        board2 = np.array([[0, 0, 0, 0],
                           [2, 4, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        td.add(board2, 2)
        td.hflip()
        expected_x = np.array([
            [[0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 4, 2], [0, 0, 0, 0], [0, 0, 0, 0]]
            ], dtype=np.int)
        expected_y = np.array([
            [0, 0, 0, 1],
            [0, 0, 1, 0]
            ], dtype=np.int)
        expected_y_digit = np.array([
            [3],
            [2]
            ], dtype=np.int)
        self.assertTrue(np.array_equal(td.get_x(), expected_x))
        self.assertTrue(np.array_equal(td.get_y(), expected_y))
        self.assertTrue(np.array_equal(td.get_y_digit(), expected_y_digit))

    def test_rotate(self):
        td = training_data.training_data()
        board1 = np.array([[1, 1, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        td.add(board1, 1)
        board2 = np.array([[0, 0, 0, 0],
                           [2, 4, 0, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        td.add(board2, 2)
        td.rotate(3)
        expected_x = np.array([
            [[0, 0, 0, 0], [0, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 4, 0, 0], [0, 2, 0, 0]]
            ], dtype=np.int)
        expected_y = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
            ], dtype=np.int)
        expected_y_digit = np.array([
            [0],
            [1]
            ], dtype=np.int)
        self.assertTrue(np.array_equal(td.get_x(), expected_x))
        self.assertTrue(np.array_equal(td.get_y(), expected_y))
        self.assertTrue(np.array_equal(td.get_y_digit(), expected_y_digit))

    def test_augment(self):
        td = training_data.training_data()
        initial_board = np.array([[1, 1, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0],
                                  [0, 0, 0, 0]])
        td.add(initial_board, 1)
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
        expected_y = np.array([
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0]
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
        self.assertTrue(np.array_equal(td.get_x(), expected_x))
        self.assertTrue(np.array_equal(td.get_y(), expected_y))
        self.assertTrue(np.array_equal(td.get_y_digit(), expected_y_digit))

    def test_merge(self):
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 1)
        td2 = training_data.training_data()
        td2.add(np.zeros([1, 4, 4]), 2)
        td.merge(td2)
        expected_x = np.array([
            [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
            [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
            ], dtype=np.int)
        expected_y = np.array([[0, 1, 0, 0], [0, 0, 1, 0]], dtype=np.int)
        self.assertTrue(np.array_equal(td.get_x(), expected_x))
        self.assertTrue(np.array_equal(td.get_y(), expected_y))

    def test_size(self):
        td = training_data.training_data()
        self.assertEqual(td.size(), 0)

if __name__ == '__main__':
    unittest.main()
