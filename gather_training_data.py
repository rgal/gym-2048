"""Gather training data from the game"""

from __future__ import print_function

import argparse
import numpy as np
import pygame

import gym

import gym_2048

def gather_training_data(env, seed=None):
    """Gather training data from letting the user play the game"""
    # Data is a list of input and outputs
    data = list()
    # Initialise seed for environment
    if seed:
        env.seed(seed)
    else:
        env.seed()
    observation = env.reset()
    print("User cursor keys to play, q to quit")
    for t in range(10):
        action = None
        env.render()
        # Ask user for action
        event = pygame.event.wait()
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = 0
            if event.key == pygame.K_RIGHT:
                action = 1
            if event.key == pygame.K_DOWN:
                action = 2
            if event.key == pygame.K_LEFT:
                action = 3
            if event.key == pygame.K_q:
                break
        # Read and discard the keyup event
        event = pygame.event.wait()
        print("Read action {}".format(action))

        # Add this data to the data collection
        data.append({'input': observation, 'output': action})
        observation, reward, done, info = env.step(action)
        print()

        if done:
            break

    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None, help="Set the seed for the game")
    args = parser.parse_args()
    # Initialise environment
    env = gym.make('2048-v0')

    # Initialise pygame for detecting keypresses
    pygame.init()

    data = gather_training_data(env, seed=args.seed)

    # Close the environment
    env.close()

    data_size = len(data)
    print("Got {} data values".format(data_size))
    x = np.empty([data_size, 16], dtype=np.int)
    for idx, d in enumerate(data):
        x[idx] = d['input']
    y = np.zeros([data_size, 4], dtype=np.int)
    for idx, d in enumerate(data):
        y[idx, d['output']] = 1

    print("Inputs: {}".format(x))
    print("Outputs: {}".format(y))

    # Save training data
    with open('x.npy', 'w') as f:
        np.save(f, x)
    with open('y.npy', 'w') as f:
        np.save(f, y)
