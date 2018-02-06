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

if __name__ == '__main__':
    unittest.main()
