from __future__ import print_function

import argparse
import csv
import json
import random
import datetime
import numpy as np

import training_data
import deep_model

def bar(value, minimum, maximum, size=50):
    """Print a bar from 0 to value, either side of 0.
    e.g. -1, -1, 1, 3 would be |||
         1, -1, 1, 3 would be    |||"""
    sections = abs(int(size * (value) / (maximum - minimum)))
    blanks = size / 2
    if value <  0.:
        blanks -= sections
    return ' ' * blanks + '|' * sections

def train(estimator, replay_memory, gamma=0.9, iterations=100, minibatch_size=64):
    """Train estimator using replay_memory"""
    #print("Training")

    for i in range(iterations):
        print("Iteration {}".format(i))
        example = np.array([[2, 0, 0, 2], [8, 8, 0, 0], [2, 16, 2, 0], [4, 4, 4, 4]]).reshape((1, 4, 4))
        prediction_2d = deep_model.get_predictions(estimator, example)
        predictions = prediction_2d.reshape((4))
        for i, v in enumerate(predictions):
            print("Action: {} Quality: {: .3f} {}".format(i, v, bar(v, -3, +3)))

        # Do the training
        # Train from minibatch from replay memory
        #print(replay_memory.size())
        sample_indexes = random.sample(range(replay_memory.size()), minibatch_size)
        #print(sample_indexes)
        sample_data = replay_memory.sample(sample_indexes)

        # Q(S, A) <- Q(S, A) + alpha(R + gamma * max(Q(S', A')) - Q(S, A)
        # Set up the target value as reward (from replay memory) + gamma * max()
        sample_rewards = sample_data.get_reward() # (batch_size, 1)
        sample_next_states = sample_data.get_next_x() # (batch_size, 4, 4)
        max_next_prediction = deep_model.get_maxq_per_state(estimator, sample_next_states)

        myu = 10.0
        sigma = 10.0
        # print("sample_rewards")
        # print(sample_rewards)
        # print("max_next_prediction")
        # print(max_next_prediction)
        # Max prediction comes out normalized so denormalize it
        max_next_prediction = max_next_prediction * sigma + myu
        # print("scaled up max_next_prediction")
        # print(max_next_prediction)
        # print("gamma * max_next_prediction")
        # print(gamma * max_next_prediction)
        target = sample_rewards + gamma * max_next_prediction
        #print(sample_rewards)
        # print("target")
        # print(target)
        # Target all at game score scale, normalize to help training
        target = (target - myu) / sigma
        # print("normalized target")
        # print(target)

        #print(sample_data.get_x())
        #print(sample_data.get_y_digit())
        #print(target)
        train_input_fn = deep_model.numpy_train_fn(sample_data.get_x(), sample_data.get_y_digit(), target)
        estimator.train(input_fn=train_input_fn)

        print("")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--input', '-i', default='data.csv', help="Data to learn from")
    parser.add_argument('--iterations', type=int, default=100, help="How many iterations to train for")
    parser.add_argument('--minibatch-size', '-m', default=64, help="Minibatch size")
    parser.add_argument('--params', '-p', default='params.json', help="Parameters file")
    parser.add_argument('--gamma', '-g', default=0.9, type=float, help="Gamma, discount factor for future rewards")
    args = parser.parse_args()

    # Load hyperparameters from file
    with open(args.params, 'r') as f:
        params = json.load(f)

    # Load estimator
    estimator = deep_model.estimator(params)

    start = datetime.datetime.now()
    replay_memory = training_data.training_data(True)
    replay_memory.import_csv(args.input)
    #replay_memory.augment()
    train(estimator, replay_memory, args.gamma, args.iterations, args.minibatch_size)

    end = datetime.datetime.now()
    taken = end - start
    print("took {}".format(taken))
