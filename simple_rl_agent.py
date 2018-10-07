from __future__ import print_function

import argparse
import random
import datetime
import numpy as np

import gym

import gym_2048

def choose_action():
    return random.randint(0, 3)

def train(seed=None, agent_seed=None):
    """seed (optional) specifies the seed for the game.
    agent_seed specifies the seed for the agent."""
    #print("Training")
    # Initialise seed for environment
    if seed:
        env.seed(seed)
    else:
        env.seed()
    # Initialise seed for agent
    if agent_seed:
        random.seed(agent_seed)
    else:
        random.seed()
    observation = env.reset()
    history = list()
    # Initialise S, A
    action = choose_action()
    illegal_count = 0
    total_reward = 0.0
    while 1:
        #env.render()
        #print(observation)
        #print "Action: {}".format(action)
        #last_observation = tuple(observation)
        #print "Observation: {}".format(last_observation)
        # Take action, observe R, S'
        next_observation, reward, done, info = env.step(action)
        total_reward += reward
        #print "New Observation: {}, reward: {}, done: {}, info: {}".format(next_observation, reward, done, info)
        # Record what we did in a particular state
        if np.array_equal(observation, next_observation):
            illegal_count += 1
            if illegal_count > 100:
                print("No progress for 100 turns, breaking out")
                break
        else:
            illegal_count = 0

        history.append((tuple(observation), action, reward))

        if done:
            break

        # Choose A' from S' using policy derived from Q
        next_action = choose_action()

        observation = next_observation
        action = next_action
        #print("")

    return total_reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1000)
    parser.add_argument('--reportfrequency', type=int, default=100)
    args = parser.parse_args()

    env = gym.make('2048-v0')
    seen_states = dict()

    start = datetime.datetime.now()
    for i_episode in range(args.episodes):
        print(train())

    # Close the environment
    env.close()

    end = datetime.datetime.now()
    taken = end - start
    print("{} episodes took {}. {:.1f} episodes per second".format(args.episodes, taken, args.episodes / taken.total_seconds()))
