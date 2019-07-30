#!/usr/bin/env python

from __future__ import absolute_import

import numpy as np
import os
import pytest
import tempfile

import training_data

class TestTrainingData():
    def test_add(self):
        td = training_data.training_data()
        assert np.array_equal(td.get_x(), np.empty([0, 4, 4], dtype=np.int))
        assert np.array_equal(td.get_y_digit(), np.empty([0, 1], dtype=np.int))
        assert np.allclose(td.get_reward(), np.empty([0, 1], dtype=np.float))
        assert np.array_equal(td.get_next_x(), np.empty([0, 4, 4], dtype=np.int))
        assert np.array_equal(td.get_done(), np.empty([0, 1], dtype=np.bool))
        td.add(np.ones([1, 4, 4]), 1, 4, np.zeros([1, 4, 4]), True)
        assert np.array_equal(td.get_x(), np.ones([1, 4, 4], dtype=np.int))
        assert np.array_equal(td.get_y_digit(), np.array([[1]], dtype=np.int))
        assert np.allclose(td.get_reward(), np.array([[4]], dtype=np.float))
        assert np.array_equal(td.get_next_x(), np.zeros([1, 4, 4], dtype=np.int))
        assert np.array_equal(td.get_done(), np.array([[1]], dtype=np.bool))

    def test_get_y_one_hot(self):
        td = training_data.training_data()
        td.add(np.ones([4, 4]), 0, 4, np.zeros([4, 4]))
        td.add(np.zeros([4, 4]), 1, 8, np.ones([4, 4]))
        td.add(np.zeros([4, 4]), 3, 8, np.ones([4, 4]))
        td.add(np.zeros([4, 4]), 2, 8, np.ones([4, 4]))
        expected_y_one_hot = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
            ], dtype=np.int)
        assert np.array_equal(td.get_y_one_hot(), expected_y_one_hot)

    def test_get_total_reward(self):
        td = training_data.training_data()
        td.add(np.ones([4, 4]), 0, 4, np.zeros([4, 4]))
        td.add(np.zeros([4, 4]), 1, 8, np.ones([4, 4]))
        td.add(np.zeros([4, 4]), 3, 16, np.ones([4, 4]))
        td.add(np.zeros([4, 4]), 2, 32, np.ones([4, 4]))
        assert td.get_total_reward() == 60

    def test_get_highest_tile(self):
        td = training_data.training_data()
        td.add(np.full((4, 4), 1), 0, 4, np.full((4, 4), 2))
        td.add(np.full((4, 4), 2), 0, 4, np.full((4, 4), 4))
        assert td.get_highest_tile() == 4

    def test_get_n(self):
        td = training_data.training_data()
        td.add(np.ones([4, 4]), 1, 4, np.zeros([4, 4]))
        td.add(np.zeros([4, 4]), 2, 8, np.ones([4, 4]))
        (state, action, reward, next_state, done) = td.get_n(1)
        assert np.array_equal(state, np.zeros([4, 4], dtype=np.int))
        assert action == 2
        assert reward == pytest.approx(8.)
        assert np.array_equal(next_state, np.ones([4, 4], dtype=np.int))

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
        assert np.array_equal(td.get_x(), expected_x)
        assert np.array_equal(td.get_y_digit(), expected_y_digit)
        assert np.allclose(td.get_reward(), expected_reward)
        assert np.allclose(td.get_next_x(), expected_next_x)

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
        assert np.array_equal(td.get_x(), expected_x)
        assert np.array_equal(td.get_y_digit(), expected_y_digit)
        assert np.allclose(td.get_reward(), expected_reward)
        assert np.array_equal(td.get_next_x(), expected_next_x)

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
        assert td.size() == 8
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
        assert np.array_equal(td.get_x(), expected_x)
        assert np.array_equal(td.get_y_digit(), expected_y_digit)
        assert np.allclose(td.get_reward(), expected_reward)
        assert np.array_equal(td.get_next_x(), expected_next_x)

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
        assert np.array_equal(td.get_x(), expected_x)
        assert np.array_equal(td.get_y_digit(), expected_y_digit)
        assert np.allclose(td.get_reward(), expected_reward)
        assert np.array_equal(td.get_next_x(), expected_next_x)

    def test_split(self):
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 1, 16, np.zeros([1, 4, 4]))
        td2 = training_data.training_data()
        td2.add(np.zeros([1, 4, 4]), 2, 0, np.ones([1, 4, 4]))
        td.merge(td2)
        a, b = td.split()
        assert np.array_equal(a.get_x(), np.ones([1, 4, 4]))
        assert np.array_equal(a.get_y_digit(), [[1]])
        assert np.array_equal(a.get_reward(), [[16]])
        assert np.array_equal(a.get_next_x(), np.zeros([1, 4, 4]))
        assert np.array_equal(b.get_x(), np.zeros([1, 4, 4]))
        assert np.array_equal(b.get_y_digit(), [[2]])
        assert np.array_equal(b.get_reward(), [[0]])
        assert np.array_equal(b.get_next_x(), np.ones([1, 4, 4]))

    def test_sample(self):
        td = training_data.training_data()
        td.add(np.zeros([1, 4, 4]), 0, 0, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 1, 1, np.ones([1, 4, 4]))
        sample = td.sample([1])
        assert sample.size() == 1
        assert sample.get_y_digit() in [[[0]], [[1]]]
        if sample.get_y_digit() == 0:
            assert np.array_equal(sample.get_x(), np.zeros([1, 4, 4]))
        if sample.get_y_digit() == 1:
            assert np.array_equal(sample.get_x(), np.ones([1, 4, 4]))

    def test_size(self):
        td = training_data.training_data()
        assert td.size() == 0
        td.add(np.ones([1, 4, 4]), 0, 4, np.zeros([1, 4, 4]))
        assert td.size() == 1

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
        assert np.allclose(td.get_reward(), expected_reward)
        expected_action = np.array([
            [0], [1], [2], [3], [0], [1]
            ], dtype=np.int)
        assert np.allclose(td.get_y_digit(), expected_action)

    def test_get_lambda_return(self):
        # Set up training data
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 0, 4, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 1, 2, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 2, 16, np.zeros([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 3, 2, np.zeros([1, 4, 4]))

        # Test using default lambda value of 0.9
        td2 = td.copy()
        lambda_return = td2.get_lambda_return()
        expected_reward = np.array([
            [20.218], [18.02], [17.8], [2.0]
            ], dtype=np.float)
        assert np.allclose(lambda_return, expected_reward)

        # Test using lambda value of 0, should have no effect on rewards
        td2 = td.copy()
        lambda_return = td2.get_lambda_return(llambda=0.0)
        expected_reward = np.array([
            [4], [2], [16], [2]
            ], dtype=np.float)
        assert np.allclose(lambda_return, expected_reward)

        # Test end of episode
        td3 = training_data.training_data()
        td3.add(np.ones([1, 4, 4]), 0, 4, np.zeros([1, 4, 4]), False)
        td3.add(np.ones([1, 4, 4]), 1, 2, np.zeros([1, 4, 4]), True)
        td3.add(np.ones([1, 4, 4]), 2, 16, np.zeros([1, 4, 4]), False)
        td3.add(np.ones([1, 4, 4]), 3, 2, np.zeros([1, 4, 4]), True)
        lambda_return = td3.get_lambda_return()
        expected_reward = np.array([
            [5.8], [2.0], [17.8], [2.0]
            ], dtype=np.float)
        assert np.allclose(lambda_return, expected_reward)

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
        assert np.allclose(td.get_reward(), expected_reward)
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
        assert np.allclose(td.get_reward(), expected_reward)

    def test_normalize_boards(self):
        # Test calculating mean and standard deviation
        td = training_data.training_data()
        td.add(np.full((1, 4, 4), 4), 1, 4, np.full((1, 4, 4), 8))
        td.add(np.full((1, 4, 4), 8), 2, 4, np.full((1, 4, 4), 16))
        td.add(np.full((1, 4, 4), 16), 3, 4, np.full((1, 4, 4), 32))
        td.add(np.full((1, 4, 4), 32), 4, 4, np.full((1, 4, 4), 64))
        td.normalize_boards()
        mean = 15.
        sd = 10.7238052947636
        a = (4. - mean) / sd
        b = (8. - mean) / sd
        c = (16. - mean) / sd
        d = (32. - mean) / sd
        e = (64. - mean) / sd
        expected_x = np.array([
            [[a, a, a, a], [a, a, a, a], [a, a, a, a], [a, a, a, a]],
            [[b, b, b, b], [b, b, b, b], [b, b, b, b], [b, b, b, b]],
            [[c, c, c, c], [c, c, c, c], [c, c, c, c], [c, c, c, c]],
            [[d, d, d, d], [d, d, d, d], [d, d, d, d], [d, d, d, d]]
            ], dtype=np.float)
        assert np.allclose(td.get_x(), expected_x)
        expected_next_x = np.array([
            [[b, b, b, b], [b, b, b, b], [b, b, b, b], [b, b, b, b]],
            [[c, c, c, c], [c, c, c, c], [c, c, c, c], [c, c, c, c]],
            [[d, d, d, d], [d, d, d, d], [d, d, d, d], [d, d, d, d]],
            [[e, e, e, e], [e, e, e, e], [e, e, e, e], [e, e, e, e]]
            ], dtype=np.float)
        assert np.allclose(td.get_next_x(), expected_next_x)

    def test_save_restore(self):
        # Set up training data
        td = training_data.training_data()
        td.add(np.ones([1, 4, 4]), 0, 4, np.zeros([1, 4, 4]))
        td.add(np.zeros([1, 4, 4]), 1, 2, np.ones([1, 4, 4]))
        td.add(np.ones([1, 4, 4]), 2, 16, np.zeros([1, 4, 4]))
        td.add(np.zeros([1, 4, 4]), 3, 2, np.ones([1, 4, 4]))

        temp_dir = tempfile.mkdtemp()
        temp_filename = os.path.join(temp_dir, 'data.csv')
        td.export_csv(temp_filename)

        td2 = training_data.training_data()
        td2.import_csv(temp_filename)

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
        assert np.array_equal(td2.get_x(), expected_x)
        assert np.array_equal(td2.get_y_digit(), expected_y_digit)
        assert np.allclose(td2.get_reward(), expected_reward)
        assert np.array_equal(td2.get_next_x(), expected_next_x)
        os.remove(temp_filename)
        os.rmdir(temp_dir)

    def test_shuffle(self):
        td = training_data.training_data()
        n = 5
        for i in range(n):
            # Use "is odd" for done
            td.add(np.full((1, 4, 4), i), i, i, np.full((1, 4, 4), i), (i % 2) == 1)
        td.shuffle()
        for i in range(n):
            # Find where this has been shuffled too
            index_of_val = np.asscalar(np.where(td.get_y_digit() == i)[0])

            # Check that all parts of this equal i
            arrays = td.get_n(index_of_val)
            for a in arrays:
                if a.dtype is np.dtype(np.bool):
                    assert((a == ((i % 2) == 1)).all())
                else:
                    assert((a == i).all())

if __name__ == '__main__':
    import pytest
    pytest.main()
