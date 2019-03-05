"""Add rewards to existing training data that doesn't have them."""

from __future__ import print_function

import argparse
import numpy as np

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
            # Read and discard the keyup event
            print("Read action {}".format(action))

            # Add this data to the data collection
            new_observation, reward, done, info = env.step(action)
            if np.array_equal(observation, new_observation):
                print("Suppressing recording of illegal move")
            else:
                data.add(observation, action, reward)
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

def get_reward_for_state_action(env, state, action):
    env.reset()
    env.set_board(state)
    new_observation, reward, done, info = env.step(action)
    return reward

def add_rewards_to_training_data(env, input_training_data):
    new_training_data = training_data.training_data()
    for n in range(input_training_data.size()):
        (state, action) = input_training_data.get_n(n)
        reward = get_reward_for_state_action(env, state, action)
        new_training_data.add(state, action, reward)
    return new_training_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', '-o', default='data.csv', help="Set the output file name")
    parser.add_argument('input', help="Specify input file name")
    args = parser.parse_args()
    # Initialise environment
    env = gym.make('2048-v0')

    input_training_data = training_data.training_data()
    input_training_data.import_csv(args.input)
    data = add_rewards_to_training_data(env, input_training_data)

    # Close the environment
    env.close()

    print("Got {} data values".format(data.size()))

    data.export_csv(args.output)
