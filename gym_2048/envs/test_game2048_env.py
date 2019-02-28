#!/usr/bin/env python

from __future__ import absolute_import
import numpy as np

import gym_2048.envs.game2048_env as game2048_env
import pytest

class TestBoard():
    def test_combine(self):
        b = game2048_env.Game2048Env()
        # Test not combining
        assert b.combine([0, 0, 0, 0]) == ([0, 0, 0, 0], 0)
        assert b.combine([2, 0, 0, 0]) == ([2, 0, 0, 0], 0)
        assert b.combine([2, 4, 0, 0]) == ([2, 4, 0, 0], 0)
        # None the same
        assert b.combine([2, 4, 8, 16]) == ([2, 4, 8, 16], 0)

        # Test combining
        # Left same same
        assert b.combine([2, 2, 8, 0]) == ([4, 8, 0, 0], 4)
        # Middle the same
        assert b.combine([4, 2, 2, 4]) == ([4, 4, 4, 0], 4)
        # Left and middle the same
        assert b.combine([2, 2, 2, 8]) == ([4, 2, 8, 0], 4)
        # Right the same
        assert b.combine([2, 8, 4, 4]) == ([2, 8, 8, 0], 8)
        # Left and right the same
        assert b.combine([2, 2, 4, 4]) == ([4, 8, 0, 0], 12)
        # Right and middle the same
        assert b.combine([2, 4, 4, 4]) == ([2, 8, 4, 0], 8)
        # All the same
        assert b.combine([4, 4, 4, 4]) == ([8, 8, 0, 0], 16)

        # Test short input
        assert b.combine([]) == ([0, 0, 0, 0], 0)
        assert b.combine([0]) == ([0, 0, 0, 0], 0)
        assert b.combine([2]) == ([2, 0, 0, 0], 0)
        assert b.combine([2, 4]) == ([2, 4, 0, 0], 0)
        assert b.combine([2, 2, 8]) == ([4, 8, 0, 0], 4)

    def test_shift(self):
        b = game2048_env.Game2048Env()
        # Shift left without combining
        assert b.shift([0, 0, 0, 0], 0) == ([0, 0, 0, 0], 0)
        assert b.shift([0, 2, 0, 0], 0) == ([2, 0, 0, 0], 0)
        assert b.shift([0, 2, 0, 4], 0) == ([2, 4, 0, 0], 0)
        assert b.shift([2, 4, 8, 16], 0) == ([2, 4, 8, 16], 0)

        # Shift left and combine
        assert b.shift([0, 2, 2, 8], 0) == ([4, 8, 0, 0], 4)
        assert b.shift([2, 2, 2, 8], 0) == ([4, 2, 8, 0], 4)
        assert b.shift([2, 2, 4, 4], 0) == ([4, 8, 0, 0], 12)

        # Shift right without combining
        assert b.shift([0, 0, 0, 0], 1) == ([0, 0, 0, 0], 0)
        assert b.shift([0, 2, 0, 0], 1) == ([0, 0, 0, 2], 0)
        assert b.shift([0, 2, 0, 4], 1) == ([0, 0, 2, 4], 0)
        assert b.shift([2, 4, 8, 16], 1) == ([2, 4, 8, 16], 0)

        # Shift right and combine
        assert b.shift([2, 2, 8, 0], 1) == ([0, 0, 4, 8], 4)
        assert b.shift([2, 2, 2, 8], 1) == ([0, 2, 4, 8], 4)
        assert b.shift([2, 2, 4, 4], 1) == ([0, 0, 4, 8], 12)

    def test_move(self):
        # Test a bunch of lines all moving at once.
        b = game2048_env.Game2048Env()
        # Test shift up
        b.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2, 8],
            [2, 2, 4, 4]]))
        assert b.move(0) == 12
        assert np.array_equal(b.get_board(), np.array([
            [4, 4, 8, 4],
            [2, 4, 2, 8],
            [0, 0, 4, 4],
            [0, 0, 0, 0]]))
        # Test shift right
        b.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2, 8],
            [2, 2, 4, 4]]))
        assert b.move(1) == 20
        assert np.array_equal(b.get_board(), np.array([
            [0, 0, 2, 4],
            [0, 0, 4, 8],
            [0, 2, 4, 8],
            [0, 0, 4, 8]]))
        # Test shift down
        b.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2, 8],
            [2, 2, 4, 4]]))
        assert b.move(2) == 12
        assert np.array_equal(b.get_board(), np.array([
            [0, 0, 0, 0],
            [0, 0, 8, 4],
            [2, 4, 2, 8],
            [4, 4, 4, 4]]))
        # Test shift left
        b.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2, 8],
            [2, 2, 4, 4]]))
        assert b.move(3) == 20
        assert np.array_equal(b.get_board(), np.array([
            [2, 4, 0, 0],
            [4, 8, 0, 0],
            [4, 2, 8, 0],
            [4, 8, 0, 0]]))

        # Test that doing the same move again (without anything added) is illegal
        with pytest.raises(game2048_env.IllegalMove):
            b.move(3)

        # Test a follow on move from the first one
        assert b.move(2) == 8 # shift down
        assert np.array_equal(b.get_board(), np.array([
            [0, 4, 0, 0],
            [2, 8, 0, 0],
            [4, 2, 0, 0],
            [8, 8, 8, 0]]))

    def test_highest(self):
        b = game2048_env.Game2048Env()
        b.set_board(np.array([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2048, 8],
            [2, 2, 4, 4]]))
        assert b.highest() == 2048

    def test_isend(self):
        b = game2048_env.Game2048Env()
        b.set_board(np.array([
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2]]))
        assert b.isend() == False
        b.set_board(np.array([
            [2, 4, 8, 16],
            [4, 8, 16, 2],
            [8, 16, 2, 4],
            [16, 2, 4, 8]]))
        assert b.isend() == True

if __name__ == '__main__':
    pytest.main()
