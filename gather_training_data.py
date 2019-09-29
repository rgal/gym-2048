"""Gather training data from the game"""

from __future__ import print_function

import argparse
import json
import time
import random

import gym
import numpy as np
import pygame
import tensorflow as tf
from tensorflow.keras.models import load_model

import matplotlib
matplotlib.use("Agg")

import matplotlib.backends.backend_agg as agg
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont

import gym_2048
import training_data

grid_size = 70

class Exiting(Exception):
    def __init__(self):
        super(Exiting, self).__init__()

def get_bar_chart(predictions, width, height):
    dpi = 100.
    fig = plt.figure(figsize=[width / dpi, height / dpi], # Inches
                       dpi=100,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
                       )
    ax = fig.gca()
    ax.set_ylim([0, 1])
    ax.bar(['up', 'right', 'down', 'left'], predictions)

    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    return raw_data

def unstack(stacked, layers=16):
  """Convert a single 4, 4, 16 stacked board state into flat 4, 4 board."""
  for i in range(layers):
    stacked[:,:,i] *= 2 ** (i + 1)
  flat = np.sum(stacked, axis=2)
  return flat

def render(observation):
    black = (0, 0, 0)
    grey = (128, 128, 128)
    blue = (0, 0, 128)
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
    }

    # Render with Pillow
    pil_board = Image.new("RGB", (grid_size * 4, grid_size * 4))
    draw = ImageDraw.Draw(pil_board)
    draw.rectangle([0, 0, 4 * grid_size, 4 * grid_size], grey)
    fnt = ImageFont.truetype('Tahoma.ttf', 30)

    for y in range(4):
      for x in range(4):
         o = observation[y][x]
         if o:
             draw.rectangle([x * grid_size, y * grid_size, (x + 1) * grid_size, (y + 1) * grid_size], tile_colour_map[o])
             (text_x_size, text_y_size) = draw.textsize(str(o), font=fnt)
             draw.text((x * grid_size + (grid_size - text_x_size) // 2, y * grid_size + (grid_size - text_y_size) // 2), str(o), font=fnt, fill=white)
             assert text_x_size < grid_size
             assert text_y_size < grid_size

    return np.asarray(pil_board).swapaxes(0, 1)

def gather_training_data(env, model, seed=None):
    """Gather training data from letting the user play the game"""
    # Data is a list of input and outputs
    data = training_data.training_data()
    # Initialise seed for environment
    if seed:
        env.seed(seed)
    else:
        env.seed()
    observation = env.reset()
    flat_observation = unstack(observation)
    print("User cursor keys to play, q to quit")
    try:
        while True:
            # Loop around performing moves
            action = None
            env.render()

            board_array = render(flat_observation)
            board_surface = pygame.surfarray.make_surface(board_array)
            screen.blit(board_surface, (0,0))

            # Get predictions from model
            if model:
                predictions = model.predict(np.reshape(observation.astype('float32'), (-1, 256))).reshape((4))
                predicted_action = np.argmax(predictions)
                #print(predictions)

                # Report predicted rewards for actions
                dir_dict = { 0: 'up', 1: 'right', 2: 'down', 3: 'left'}
                dir_reward = [(dir_dict[i], p) for i, p in enumerate(list(predictions))]
                dir_reward.sort(key=lambda x: x[1], reverse=True)
                for direction, reward in dir_reward:
                    print('{}: {:.3f}'.format(direction, reward))

                chart_height = 4 * grid_size
                chart_width = 4 * grid_size
                raw_data = get_bar_chart(predictions, chart_height, chart_width)

                surf = pygame.image.fromstring(raw_data, (chart_height, chart_width), "RGB")
                screen.blit(surf, (4 * grid_size, 0))

            pygame.display.update()

            # Ask user for action
            record_action = False
            while True:
                # Loop waiting for valid input
                event = pygame.event.wait()
                if event.type == pygame.KEYDOWN:
                    key_action_map = {
                        pygame.K_UP: 0,
                        pygame.K_RIGHT: 1,
                        pygame.K_DOWN: 2,
                        pygame.K_LEFT: 3
                    }
                    if event.key in key_action_map:
                        action = key_action_map[event.key]
                        record_action = True
                        break
                    if event.key == pygame.K_q:
                        raise Exiting
                    if model and (event.key == pygame.K_a):
                        # Auto-select best action according to model
                        action = predicted_action
                        break
                    if event.key == pygame.K_r:
                        # Randomly select action
                        action = random.choice([0, 1, 2, 3])
                        break
                if event.type == pygame.QUIT:
                    raise Exiting

            print("Selected action {}".format(action))

            # Add this data to the data collection if manually entered and not illegal
            new_observation, reward, done, info = env.step(action)
            # For now just unstack the stacked state
            new_flat_observation = unstack(new_observation)
            illegal_move = info['illegal_move']
            if record_action and not illegal_move:
                data.add(flat_observation, action, reward, new_flat_observation, done)
            else:
                print("Not recording move")

            observation = new_observation
            flat_observation = new_flat_observation
            print()

            if done:
                # Draw final board
                env.render()
                print("End of game")
                break
    except Exiting:
        print("Exiting...")

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default=None, help="Input trained model to get predictions from (optional)")
    parser.add_argument('--output', '-o', default='data_{}.csv'.format(int(time.time())), help="Set the output file name")
    parser.add_argument('--seed', type=int, default=None, help="Set the seed for the game")
    args = parser.parse_args()
    # Initialise environment
    env = gym.make('2048-v0')

    # Load model
    if args.input:
        model = load_model(args.input)
    else:
        model = None

    # Initialise pygame for detecting keypresses
    pygame.init()
    height = 4 * grid_size
    width = 4 * grid_size
    if model:
        width = 8 * grid_size
    screen = pygame.display.set_mode((width, height), 0, 32)
    pygame.font.init()
    data = gather_training_data(env, model, seed=args.seed)

    # Close the environment
    env.close()

    print("Got {} data values".format(data.size()))

    if data.size():
        data.export_csv(args.output)
