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

import gym_2048
import training_data
import train_keras_model

grid_size = 70

class EndingEpisode(Exception):
    def __init__(self):
        super(EndingEpisode, self).__init__()

class Quitting(Exception):
    def __init__(self):
        super(Quitting, self).__init__()

def get_figure(width, height):
    dpi = 100.
    return plt.figure(figsize=[width / dpi, height / dpi], # Inches
                      dpi=100,        # 100 dots per inch, so the resulting buffer is 400x400 pixels
                     )

def get_bar_chart(fig, predictions):
    fig.clf()
    ax = fig.gca()
    ax.set_xlabel('Action')
    ax.set_ylabel('Probability')
    ax.set_ylim([0, 1])
    ax.bar(['Up', 'Right', 'Down', 'Left'], predictions)

    plt.tight_layout()
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    return raw_data

def get_line_plot(fig, results):
    fig.clf()
    ax = fig.gca()
    ax.set_xlabel('Episode')
    ax.set_ylabel('Score')
    ax.set_xlim([0, len(results)])
    ax.plot(range(len(results)), [r['Average score'] for r in results], label="Average score")
    ax.plot(range(len(results)), [r['Max score'] for r in results], label="Max score")
    ax.legend()

    plt.tight_layout()
    canvas = agg.FigureCanvasAgg(fig)
    canvas.draw()
    renderer = canvas.get_renderer()
    raw_data = renderer.tostring_rgb()
    return raw_data

def unstack(stacked, layers=16):
    """Convert a single 4, 4, 16 stacked board state into flat 4, 4 board."""
    representation = 2 ** (np.arange(layers, dtype=int) + 1)
    return np.sum(stacked * representation, axis=2)

def high_tile_in_corner(board):
    """Reports whether the a high tile >=64 is in the corner of (flat) board."""
    assert board.shape == (4, 4)
    highest_tile = np.amax(board)
    if highest_tile < 64:
        return False
    tiles_equal_to_highest = np.equal(board, np.full((4, 4), highest_tile))
    corners_equal_to_highest = tiles_equal_to_highest[[0, 0, -1, -1], [0, -1, 0, -1]]
    high_tile_in_corner = np.any(corners_equal_to_highest)
    #print(f"{board}, {highest_tile}, {tiles_equal_to_highest}, {corners_equal_to_highest}, {high_tile_in_corner}")
    return high_tile_in_corner

def gather_training_data(env, model, data, results, seed=None):
    """Gather training data from letting the user play the game"""
    # Initialise seed for environment
    if seed:
        env.seed(seed)
    else:
        env.seed()
    observation = env.reset()
    chart_height = 4 * grid_size
    chart_width = 4 * grid_size
    fig = get_figure(chart_width, chart_height)
    fig2 = get_figure(chart_width, chart_height)
    print("User cursor keys to play, q to quit")
    try:
        while True:
            # Loop around performing moves
            action = None
            env.render()

            board_array = env.render(mode='rgb_array')
            board_surface = pygame.surfarray.make_surface(board_array)
            screen.blit(board_surface, (0, 0))

            # Get predictions from model
            predictions = model.predict(np.reshape(observation.astype('float32'), (-1, 256))).reshape((4))
            predicted_action = np.argmax(predictions)
            #print(predictions)

            # Report predicted rewards for actions
            dir_dict = {0: 'up', 1: 'right', 2: 'down', 3: 'left'}
            dir_reward = [(dir_dict[i], p) for i, p in enumerate(list(predictions))]
            dir_reward.sort(key=lambda x: x[1], reverse=True)
            for direction, reward in dir_reward:
                print('{}: {:.3f}'.format(direction, reward))

            # Create graph of predictions
            raw_data = get_bar_chart(fig, predictions)
            surf = pygame.image.fromstring(raw_data, (chart_height, chart_width), "RGB")
            screen.blit(surf, (4 * grid_size, 0))

            # Create graph of results
            raw_data2 = get_line_plot(fig2, results)
            surf2 = pygame.image.fromstring(raw_data2, (chart_height, chart_width), "RGB")
            screen.blit(surf2, (8 * grid_size, 0))

            pygame.display.update()

            # Ask user for action
            record_action = False
            # Auto-select best action according to model
            # Require at least 50% confidence
            # Naive view of confidence, not counting symmetrical boards
            confidence = np.max(predictions)
            if confidence < 0.5:
                print("***Confidence < 50%: {}***".format(confidence))

            predicted_is_illegal = False
            env2 = gym.make('2048-v0')
            env2.set_board(unstack(observation))
            (board2, _, _, info2) = env2.step(predicted_action)
            predicted_is_illegal = info2['illegal_move']
            if predicted_is_illegal:
                print("***Predicted is illegal.***")

            high_in_corner_before = high_tile_in_corner(unstack(observation))
            high_in_corner_after = high_tile_in_corner(unstack(board2))
            lost_high_corner = high_in_corner_before and not high_in_corner_after
            if lost_high_corner:
                print("***Lost high corner tile.***")

            if confidence < 0.5 or predicted_is_illegal or lost_high_corner:
                # Ask user for input
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
                        if event.key == pygame.K_e:
                            raise EndingEpisode
                        if event.key == pygame.K_q:
                            raise Quitting
                        if event.key == pygame.K_a:
                            # Auto-select best action according to model
                            action = predicted_action
                            break
                        if event.key == pygame.K_r:
                            # Randomly select action
                            action = random.randrange(4)
                            break
                    if event.type == pygame.QUIT:
                        raise Quitting
            else:
                action = predicted_action

            print("Selected action {}".format(action))

            # Add this data to the data collection if manually entered and not illegal
            new_observation, reward, done, info = env.step(action)
            illegal_move = info['illegal_move']
            if record_action and not illegal_move:
                # Unstack the stacked state
                data.add(unstack(observation), action, reward, unstack(new_observation), done)
            else:
                print("Not recording move")

            observation = new_observation
            print()

            if done:
                # Draw final board
                env.render()
                print("End of game")
                break
    except EndingEpisode:
        print("Ending episode...")

    return data


board_size = 4
board_layers = 16 # Layers of game board to represent different numbers
outputs = 4

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', '-i', default=None, help="Input existing training data to start from (optional)")
    parser.add_argument('--model', '-m', default=None, help="Pre-trained model to start from (optional)")
    parser.add_argument('--reload-results', default=None, help="Reload previous results")

    timestamp = int(time.time())
    parser.add_argument('--output', '-o', default='data_{}.csv'.format(timestamp), help="Set the output training data file name")
    parser.add_argument('--output-model', default='model_{}.hdf5'.format(timestamp), help="Set the output model file name")
    parser.add_argument('--results', '-r', default='results_{}.json'.format(timestamp), help="Set the output results file name")

    parser.add_argument('--seed', type=int, default=None, help="Set the seed for the game")
    args = parser.parse_args()
    # Initialise environment
    env = gym.make('2048-v0')

    if args.model:
        model = load_model(args.model)
    else:
        filters = 64
        residual_blocks = 8
        model = train_keras_model.build_model(board_size, board_layers, outputs, filters, residual_blocks)
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    # Initialise pygame for detecting keypresses
    pygame.init()
    height = 4 * grid_size
    width = 12 * grid_size
    screen = pygame.display.set_mode((width, height), 0, 32)
    pygame.font.init()
    alldata = training_data.training_data()
    if args.input:
        alldata.import_csv(args.input)
        train_from_me = alldata.copy()
        train_from_me.augment()
        #train_from_me.make_boards_unique()
        train_data = np.reshape(train_from_me.get_x_stacked().astype('float'), (-1, board_size * board_size * board_layers))
        train_labels = train_from_me.get_y_digit()

        model.fit(train_data,
          train_labels,
          epochs=3,
          batch_size=128)

    if args.reload_results:
        with open(args.reload_results) as r:
            results = json.load(r)
    else:
        results = [train_keras_model.evaluate_model(model, 10, 0.)]

    try:
        while True:
            gather_training_data(env, model, alldata, results, seed=args.seed)


            train_from_me = alldata.copy()
            train_from_me.augment()
            # train_from_me.make_boards_unique()
            train_data = np.reshape(train_from_me.get_x_stacked().astype('float'), (-1, board_size * board_size * board_layers))
            train_labels = train_from_me.get_y_digit()

            model.fit(train_data,
              train_labels,
              epochs=3,
              batch_size=128)

            results.append(train_keras_model.evaluate_model(model, 10, 0.))

            print("Got {} data values".format(alldata.size()))

    except Quitting:
        print("Quitting...")

    # Close the environment
    env.close()

    print(results)
    if results:
        with open(args.results, 'w') as r:
            json.dump(results, r, indent=4)

    if alldata.size():
        alldata.export_csv(args.output)

    if args.output_model:
        model.save(args.output_model)
