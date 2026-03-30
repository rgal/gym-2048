#!/usr/bin/env python

from __future__ import absolute_import
import numpy as np

import env.envs.game2048_env as game2048_env
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
        assert b.shift([0, 0, 0, 0]) == ([0, 0, 0, 0], 0)
        assert b.shift([0, 2, 0, 0]) == ([2, 0, 0, 0], 0)
        assert b.shift([0, 2, 0, 4]) == ([2, 4, 0, 0], 0)
        assert b.shift([2, 4, 8, 16]) == ([2, 4, 8, 16], 0)

        # Shift left and combine
        assert b.shift([0, 2, 2, 8]) == ([4, 8, 0, 0], 4)
        assert b.shift([2, 2, 2, 8]) == ([4, 2, 8, 0], 4)
        assert b.shift([2, 2, 4, 4]) == ([4, 8, 0, 0], 12)

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

        # Full board with adjacent equal tiles — legal moves exist
        b.set_board(np.array([
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2],
            [2, 2, 2, 2]]))
        assert b.isend() == False

        # Full board with no adjacent equals — no legal moves
        b.set_board(np.array([
            [2, 4, 8, 16],
            [4, 8, 16, 2],
            [8, 16, 2, 4],
            [16, 2, 4, 8]]))
        assert b.isend() == True

        # Board with empty cells — must return False via early exit
        b.set_board(np.array([
            [2, 4, 8, 16],
            [4, 8, 16, 2],
            [8, 16, 2, 4],
            [16, 2, 4, 0]]))
        assert b.isend() == False

        # max_tile reached — must return True regardless of empty cells
        b.set_max_tile(2048)
        b.set_board(np.array([
            [2048, 0, 0, 0],
            [0,    0, 0, 0],
            [0,    0, 0, 0],
            [0,    0, 0, 0]]))
        assert b.isend() == True

        # max_tile set but not yet reached — empty cells present, should be False
        b.set_board(np.array([
            [1024, 0, 0, 0],
            [0,    0, 0, 0],
            [0,    0, 0, 0],
            [0,    0, 0, 0]]))
        assert b.isend() == False

class TestStep():
    def test_step_returns_correct_shapes(self):
        b = game2048_env.Game2048Env()
        b.reset(seed=0)
        obs, reward, terminated, truncated, info = b.step(0)
        assert obs.shape == (16, 4, 4)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert 'illegal_move' in info
        assert 'highest' in info

    def test_step_reward_equals_merge_score(self):
        b = game2048_env.Game2048Env()
        b.reset(seed=0)
        # Set up a board where merging is guaranteed: two 2s in the same column
        b.set_board(np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0]]))
        obs, reward, terminated, truncated, info = b.step(0)  # shift up
        assert reward == 4.0

    def test_step_score_accumulates(self):
        b = game2048_env.Game2048Env()
        b.reset(seed=0)
        b.set_board(np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [2, 0, 0, 0],
            [2, 0, 0, 0]]))
        b.step(0)  # merges two 2s → score 4
        b.set_board(np.array([
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [4, 0, 0, 0],
            [4, 0, 0, 0]]))
        b.step(0)  # merges two 4s → score 8
        assert b.score == 12.0

    def test_step_illegal_move_terminates(self):
        b = game2048_env.Game2048Env()
        b.reset(seed=0)
        # Board that cannot shift up (all tiles already at top or blocked)
        b.set_board(np.array([
            [2, 4, 8, 16],
            [4, 8, 16, 2],
            [8, 16, 2, 4],
            [16, 2, 4, 8]]))
        obs, reward, terminated, truncated, info = b.step(0)
        assert terminated == True
        assert info['illegal_move'] == True

    def test_step_illegal_move_reward(self):
        b = game2048_env.Game2048Env()
        b.set_illegal_move_reward(-1.0)
        b.reset(seed=0)
        b.set_board(np.array([
            [2, 4, 8, 16],
            [4, 8, 16, 2],
            [8, 16, 2, 4],
            [16, 2, 4, 8]]))
        obs, reward, terminated, truncated, info = b.step(0)
        assert reward == -1.0

    def test_step_observation_is_valid_one_hot(self):
        b = game2048_env.Game2048Env()
        b.reset(seed=0)
        b.set_board(np.array([
            [2, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 4, 0]]))
        obs, _, _, _, _ = b.step(1)  # shift right
        # Each cell should be one-hot: at most one channel set per position
        assert obs.shape == (16, 4, 4)
        assert obs.sum(axis=0).max() <= 1
        assert set(obs.flatten().tolist()) == {0, 1}


if __name__ == '__main__':
    pytest.main()
