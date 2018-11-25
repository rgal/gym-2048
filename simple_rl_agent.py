from __future__ import print_function

import argparse
import csv
import json
import random
import datetime
import numpy as np

import gym

import gym_2048

import training_data
import deep_model

last_observation = None
last_chosen = None

def bar(value, minimum, maximum, size=20):
   """Print a bar from minimum to value"""
   sections = int(size * (value - minimum) / (maximum - minimum))
   return '|' * sections

def get_prediction(observation):
    predict_input_fn = deep_model.numpy_predict_fn(observation)
    prediction = list(estimator.predict(input_fn=predict_input_fn))[0]
    return prediction['logits']

def choose_action(estimator, observation, epsilon=0.1):
    """Choose best action from the esimator or random, based on epsilon"""
    global last_observation
    global last_chosen
    if (random.uniform(0, 1) > epsilon):
        if last_observation is not None and np.array_equal(last_observation, observation):
            print("Returning cached best action: {}".format(last_chosen))
            return last_chosen
        prediction = get_prediction(observation)
        print(prediction)
        for i, v in enumerate(prediction):
            print("Action: {} Quality: {}".format(i, bar(v, -3, +3)))
        chosen = np.argmax(prediction)
        print("Choosing best action: {}".format(chosen))
        # Update last chosen
        last_chosen = chosen
        last_observation = observation
    else:
        chosen = random.randint(0, 3)
        print("Choosing random action: {}".format(chosen))
    return chosen

def train(estimator, epsilon, seed=None, agent_seed=None):
    """Train estimator for one episode.
    seed (optional) specifies the seed for the game.
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
    data = training_data.training_data()

    illegal_count = 0
    total_reward = 0.0
    total_illegals = 0
    moves_taken = 0
    while 1:
        #env.render()
        #print(observation)
        #print "Action: {}".format(action)
        #last_observation = tuple(observation)
        #print "Observation: {}".format(last_observation)
        # Take action, observe R, S'
        print(observation.reshape((4, 4)))
        print("Score: {}".format(total_reward))
        action = choose_action(estimator, observation, epsilon)
        next_observation, reward, done, info = env.step(action)
        total_reward += reward
        #print "New Observation: {}, reward: {}, done: {}, info: {}".format(next_observation, reward, done, info)
        # Record what we did in a particular state
        if np.array_equal(observation, next_observation):
            print("Illegal move selected {}".format(illegal_count))
            total_illegals += 1
            illegal_count += 1
            if illegal_count > 100:
                print("No progress for 100 turns, breaking out")
                break
        else:
            illegal_count = 0

        data.add(observation, action, reward)
        moves_taken += 1

        if done:
            break

        observation = next_observation
        print("")

    print("Took {} moves, Score: {}".format(moves_taken, total_reward))
    # Apply training to model using history of boards, actions and total reward
    data.smooth_rewards()

    # Mean and SD derived from supervised learning data
    data.normalize_rewards(mean=175, sd=178)

    # Augment data
    # This could make it harder to learn but will make more of the data we have
    # and it is quicker to learn in larger batches
    data.augment()

    train_input_fn = deep_model.numpy_train_fn(data.get_x(), data.get_y_digit(), data.get_reward())
    estimator.train(input_fn=train_input_fn)
    return data, total_reward, env.score, moves_taken, total_illegals

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=0.1, help="Probability of choosing random action instead of greedy")
    parser.add_argument('--params', '-p', default='params.json', help="Parameters file")
    parser.add_argument('--output', default='dat.csv')
    args = parser.parse_args()

    env = gym.make('2048-v0')
    env = env.unwrapped
    env.set_illegal_move_reward(-1.)

    # Load hyperparameters from file
    with open(args.params, 'r') as f:
        params = json.load(f)

    # Load estimator
    estimator = deep_model.estimator(params)

    start = datetime.datetime.now()
    scores = list()
    all_data = training_data.training_data()
    for i_episode in range(args.episodes):
        print("Episode {}".format(i_episode))
        (data, total_reward, score, moves_taken, illegal_count) = train(estimator, args.epsilon)
        scores.append({'score': score, 'total_reward': total_reward, 'moves': moves_taken, 'illegal_count': illegal_count, 'highest': env.highest()})
        #print(score)
        all_data.merge(data)

    print(scores)
    with open('scores.csv', 'w') as f:
        w = csv.DictWriter(f, ['score', 'total_reward', 'moves', 'illegal_count', 'highest'])
        w.writeheader()
        for s in scores:
            w.writerow(s)

    print("Average: {}".format(np.mean([s['score'] for s in scores])))
    # Close the environment
    env.close()

    end = datetime.datetime.now()
    taken = end - start
    print("{} episodes took {}. {:.1f} episodes per second".format(args.episodes, taken, args.episodes / taken.total_seconds()))
