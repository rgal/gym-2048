"""Gather training data from the game"""

from __future__ import print_function

import argparse
import numpy as np
import os
import pygame

import gym

import gym_2048

import training_data

class Exiting(Exception):
    def __init__(self):
        super(Exiting, self).__init__()

def gather_training_data(env, seed=None):
    """Gather training data from letting the user play the game"""
    # Data is a list of input and outputs
    data = training_data.training_data()
    # Initialise seed for environment
    if seed:
        env.seed(seed)
    else:
        env.seed()
    observation = env.reset()
    print("User cursor keys to play, q to quit")
    try:
        while True:
            # Loop around performing moves
            action = None
            env.render()
            # Ask user for action
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
                        break
                    if event.key == pygame.K_q:
                        raise Exiting
                if event.type == pygame.QUIT:
                    raise Exiting
            # Read and discard the keyup event
            print("Read action {}".format(action))

            # Add this data to the data collection
            new_observation, reward, done, info = env.step(action)
            if np.array_equal(observation, new_observation):
                print("Suppressing recording of illegal move")
            else:
                data.add(observation, action)
            observation = new_observation
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
    parser.add_argument('--output', '-o', default='data.csv', help="Set the output file name")
    parser.add_argument('--seed', type=int, default=None, help="Set the seed for the game")
    args = parser.parse_args()
    # Initialise environment
    env = gym.make('2048-v0')

    # Initialise pygame for detecting keypresses
    pygame.init()

    data = gather_training_data(env, seed=args.seed)

    # Close the environment
    env.close()

    print("Got {} data values".format(data.size()))

    data.export_csv(args.output)
