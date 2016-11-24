#!/usr/bin/env python

import unittest

import game2048_env

class TestBoard(unittest.TestCase):
    def test_combine(self):
        b = game2048_env.Game2048Env()
        # Test not combining
        self.assertEqual(b.combine([0, 0, 0, 0]), ([0, 0, 0, 0], 0))
        self.assertEqual(b.combine([2, 0, 0, 0]), ([2, 0, 0, 0], 0))
        self.assertEqual(b.combine([2, 4, 0, 0]), ([2, 4, 0, 0], 0))
        self.assertEqual(b.combine([2, 4, 8, 16]), ([2, 4, 8, 16], 0))

        # Test combining
        self.assertEqual(b.combine([2, 2, 8, 0]), ([4, 8, 0, 0], 4))
        self.assertEqual(b.combine([2, 2, 2, 8]), ([4, 2, 8, 0], 4))
        self.assertEqual(b.combine([2, 2, 4, 4]), ([4, 8, 0, 0], 12))
        self.assertEqual(b.combine([4, 2, 2, 4]), ([4, 4, 4, 0], 4))
        self.assertEqual(b.combine([4, 4, 4, 4]), ([8, 8, 0, 0], 16))

    def test_shift(self):
        b = game2048_env.Game2048Env()
        # Shift left without combining
        self.assertEqual(b.shift([0, 0, 0, 0], 0), ([0, 0, 0, 0], 0))
        self.assertEqual(b.shift([0, 2, 0, 0], 0), ([2, 0, 0, 0], 0))
        self.assertEqual(b.shift([0, 2, 0, 4], 0), ([2, 4, 0, 0], 0))
        self.assertEqual(b.shift([2, 4, 8, 16], 0), ([2, 4, 8, 16], 0))

        # Shift left and combine
        self.assertEqual(b.shift([0, 2, 2, 8], 0), ([4, 8, 0, 0], 4))
        self.assertEqual(b.shift([2, 2, 2, 8], 0), ([4, 2, 8, 0], 4))
        self.assertEqual(b.shift([2, 2, 4, 4], 0), ([4, 8, 0, 0], 12))

        # Shift right without combining
        self.assertEqual(b.shift([0, 0, 0, 0], 1), ([0, 0, 0, 0], 0))
        self.assertEqual(b.shift([0, 2, 0, 0], 1), ([0, 0, 0, 2], 0))
        self.assertEqual(b.shift([0, 2, 0, 4], 1), ([0, 0, 2, 4], 0))
        self.assertEqual(b.shift([2, 4, 8, 16], 1), ([2, 4, 8, 16], 0))

        # Shift right and combine
        self.assertEqual(b.shift([2, 2, 8, 0], 1), ([0, 0, 4, 8], 4))
        self.assertEqual(b.shift([2, 2, 2, 8], 1), ([0, 2, 4, 8], 4))
        self.assertEqual(b.shift([2, 2, 4, 4], 1), ([0, 0, 4, 8], 12))

    def test_move(self):
        # Test a bunch of lines all moving at once.
        b = game2048_env.Game2048Env()
        b.set_board([
            [0, 2, 0, 4],
            [2, 2, 8, 0],
            [2, 2, 2, 8],
            [2, 2, 4, 4]])
        self.assertEqual(b.move(0), 20) # shift to the left
        self.assertEqual(b.get_board(), [
            [2, 4, 0, 0],
            [4, 8, 0, 0],
            [4, 2, 8, 0],
            [4, 8, 0, 0]])

        # Test that doing the same move again (without anything added) is illegal
        with self.assertRaises(game2048_env.IllegalMove):
            b.move(0)

if __name__ == '__main__':
    unittest.main()
