from __future__ import print_function

import gymnasium as gym
from gymnasium import spaces
from gymnasium.utils import seeding

import numpy as np

from PIL import Image, ImageDraw, ImageFont

import logging
import random
from io import StringIO
import sys

class IllegalMove(Exception):
    pass

def stack(flat, layers=16):
  """Convert an [4, 4] representation into [layers, 4, 4] with one layer for each value."""
  # representation is what each layer represents
  representation = 2 ** (np.arange(layers, dtype=int) + 1)

  # layered is the flat board repeated layers times (channels-first)
  layered = np.repeat(flat[np.newaxis, :, :], layers, axis=0)

  # Now set the values in the board to 1 or zero depending whether they match representation.
  # Representation is broadcast across H and W axes
  layered = np.where(layered == representation[:, np.newaxis, np.newaxis], 1, 0)

  return layered

class Game2048Env(gym.Env):
    metadata = {'render_modes': ['ansi', 'human', 'rgb_array'], 'render_fps': 4}
    _all_positions = [(r, c) for r in range(4) for c in range(4)]

    def __init__(self, render_mode: str | None = None):
        # Definitions for game. Board must be square.
        self.size = 4
        self.w = self.size
        self.h = self.size
        self.squares = self.size * self.size

        # Maintain own idea of game score, separate from rewards
        self.score = 0

        # Members for gym implementation
        self.action_space = spaces.Discrete(4)
        # Suppose that the maximum tile is as if you have powers of 2 across the board.
        layers = self.squares
        self.observation_space = spaces.Box(0, 1, (layers, self.w, self.h), dtype=int)
        self.set_illegal_move_reward(0.)
        self.set_max_tile(None)

        # Size of square for rendering
        self.grid_size = 70
        self.render_mode = render_mode

    def seed(self, seed=None):
        # Keep for backward compatibility with scripts that call env.seed()
        self._np_random, seed = seeding.np_random(seed)
        return [seed]

    def set_illegal_move_reward(self, reward):
        """Define the reward/penalty for performing an illegal move. Also need
            to update the reward range for this."""
        # Guess that the maximum reward is also 2**squares though you'll probably never get that.
        # (assume that illegal move reward is the lowest value that can be returned
        self.illegal_move_reward = reward
        self.reward_range = (self.illegal_move_reward, float(2**self.squares))

    def set_max_tile(self, max_tile):
        """Define the maximum tile that will end the game (e.g. 2048). None means no limit.
           This does not affect the state returned."""
        assert max_tile is None or isinstance(max_tile, int)
        self.max_tile = max_tile

    # Implement gym interface
    def step(self, action):
        """Perform one step of the game. This involves moving and adding a new tile."""
        logging.debug("Action {}".format(action))
        score = 0
        terminated = None
        info = {
            'illegal_move': False,
        }
        try:
            score = float(self.move(action))
            self.score += score
            assert score <= 2**(self.w*self.h)
            self.add_tile()
            terminated = self.isend()
            reward = float(score)
        except IllegalMove:
            logging.debug("Illegal move")
            info['illegal_move'] = True
            terminated = True
            reward = self.illegal_move_reward

        info['highest'] = self.highest()

        # Return observation (board state), reward, terminated, truncated and info dict
        return stack(self.Matrix), reward, terminated, False, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.Matrix = np.zeros((self.h, self.w), int)
        self.score = 0

        logging.debug("Adding tiles")
        self.add_tile()
        self.add_tile()

        return stack(self.Matrix), {}

    def render(self, mode=None):
        if mode is None:
            mode = self.render_mode or 'human'
        if mode == 'rgb_array':
            black = (0, 0, 0)
            grey = (128, 128, 128)
            white = (255, 255, 255)
            tile_colour_map = {
                2: (255, 0, 0),
                4: (224, 32, 0),
                8: (192, 64, 0),
                16: (160, 96, 0),
                32: (128, 128, 0),
                64: (96, 160, 0),
                128: (64, 192, 0),
                256: (32, 224, 0),
                512: (0, 255, 0),
                1024: (0, 224, 32),
                2048: (0, 192, 64),
                4096: (0, 160, 96),
            }
            grid_size = self.grid_size

            # Render with Pillow
            pil_board = Image.new("RGB", (grid_size * 4, grid_size * 4))
            draw = ImageDraw.Draw(pil_board)
            draw.rectangle([0, 0, 4 * grid_size, 4 * grid_size], grey)
            fnt = ImageFont.truetype('Arial.ttf', 30)

            for y in range(4):
              for x in range(4):
                 o = self.get(y, x)
                 if o:
                     draw.rectangle([x * grid_size, y * grid_size, (x + 1) * grid_size, (y + 1) * grid_size], tile_colour_map[o])
                     bbox = draw.textbbox((0, 0), str(o), font=fnt)
                     text_x_size = bbox[2] - bbox[0]
                     text_y_size = bbox[3] - bbox[1]
                     draw.text((x * grid_size + (grid_size - text_x_size) // 2, y * grid_size + (grid_size - text_y_size) // 2), str(o), font=fnt, fill=white)
                     assert text_x_size < grid_size
                     assert text_y_size < grid_size

            return np.asarray(pil_board).swapaxes(0, 1)

        outfile = StringIO() if mode == 'ansi' else sys.stdout
        s = 'Score: {}\n'.format(self.score)
        s += 'Highest: {}\n'.format(self.highest())
        npa = np.array(self.Matrix)
        grid = npa.reshape((self.size, self.size))
        s += "{}\n".format(grid)
        outfile.write(s)
        return outfile

    # Implement 2048 game
    def add_tile(self):
        """Add a tile, probably a 2 but maybe a 4"""
        val = 2 if random.random() < 0.9 else 4
        positions = self._all_positions.copy()
        random.shuffle(positions)
        for r, c in positions:
            if self.Matrix[r, c] == 0:
                logging.debug("Adding %s at %s", val, (r, c))
                self.set(r, c, val)
                return
        assert False, "No empty cell found"

    def get(self, x, y):
        """Return the value of one square."""
        return self.Matrix[x, y]

    def set(self, x, y, val):
        """Set the value of one square."""
        self.Matrix[x, y] = val

    def empties(self):
        """Return a 2d numpy array with the location of empty squares."""
        return np.argwhere(self.Matrix == 0)

    def highest(self):
        """Report the highest tile on the board."""
        return np.max(self.Matrix)

    def move(self, direction, trial=False):
        """Perform one move of the game. Shift things to one side then,
        combine. directions 0, 1, 2, 3 are up, right, down, left.
        Returns the score that [would have] got."""
        if not trial:
            if direction == 0:
                logging.debug("Up")
            elif direction == 1:
                logging.debug("Right")
            elif direction == 2:
                logging.debug("Down")
            elif direction == 3:
                logging.debug("Left")

        changed = False
        move_score = 0
        dir_div_two = int(direction / 2)
        dir_mod_two = int(direction % 2)
        shift_direction = dir_mod_two ^ dir_div_two # 0 for towards up left, 1 for towards bottom right

        if dir_mod_two == 0:
            # Up or down, split into columns
            for y in range(self.h):
                old = self.Matrix[:, y].tolist()
                if shift_direction:
                    old = old[::-1]
                (new, ms) = self.shift(old)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        self.Matrix[:, y] = new[::-1] if shift_direction else new
        else:
            # Left or right, split into rows
            for x in range(self.w):
                old = self.Matrix[x, :].tolist()
                if shift_direction:
                    old = old[::-1]
                (new, ms) = self.shift(old)
                move_score += ms
                if old != new:
                    changed = True
                    if not trial:
                        self.Matrix[x, :] = new[::-1] if shift_direction else new
        if changed != True:
            raise IllegalMove

        return move_score

    def shift(self, row):
        """Compact and combine a row leftward in a single pass."""
        move_score = 0
        combined_row = [0] * self.size
        output_index = 0
        can_merge = False
        for val in row:
            if val == 0:
                continue
            if can_merge and combined_row[output_index - 1] == val:
                combined_row[output_index - 1] *= 2
                move_score += combined_row[output_index - 1]
                can_merge = False
            else:
                combined_row[output_index] = val
                output_index += 1
                can_merge = True
        return (combined_row, move_score)

    def isend(self):
        """Has the game ended. Game ends if there is a tile equal to the limit
           or there are no legal moves. If there are empty spaces then there
           must be legal moves."""

        if self.max_tile is not None and self.highest() == self.max_tile:
            return True

        if (self.Matrix == 0).any():
            return False

        for direction in range(4):
            try:
                self.move(direction, trial=True)
                # Not the end if we can do any move
                return False
            except IllegalMove:
                pass
        return True

    def get_board(self):
        """Retrieve the whole board, useful for testing."""
        return self.Matrix

    def set_board(self, new_board):
        """Retrieve the whole board, useful for testing."""
        self.Matrix = new_board
