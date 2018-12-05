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

def bar(value, minimum, maximum, size=20):
    """Print a bar from minimum to value"""
    sections = int(size * (value - minimum) / (maximum - minimum))
    return '|' * sections

def get_prediction(observation):
    predict_input_fn = deep_model.numpy_predict_fn(np.tile(observation, (4, 1)), np.arange(4))
    prediction = estimator.predict(input_fn=predict_input_fn)
    prediction_values = [np.asscalar(p['logits']) for p in prediction]
    return prediction_values

def choose_action(estimator, observation, epsilon=0.1):
    """Choose best action from the esimator or random, based on epsilon
       Return both the action id and the estimated quality."""
    prediction = get_prediction(observation)
    for i, v in enumerate(prediction):
        print("Action: {} Quality: {:.3f} {}".format(i, v, bar(v, -3, +3)))
    if random.uniform(0, 1) > epsilon:
        chosen = np.argmax(prediction)
        print("Choosing best action: {}".format(chosen))
        return chosen, np.max(prediction)
    else:
        chosen = random.randint(0, 3)
        print("Choosing random action: {}".format(chosen))
        return chosen, prediction[chosen]

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

    illegal_count = 0
    total_reward = 0.0
    total_illegals = 0
    moves_taken = 0

    # Initialise S
    state = env.reset()
    # Choose A from S using policy derived from Q
    (action, qual) = choose_action(estimator, state, epsilon)
    while 1:
        # Take action, observe R, S'
        next_state, reward, done, info = env.step(action)
        print(next_state.reshape(4, 4))
        total_reward += reward
        print("Score: {} Total reward: {}".format(env.score, total_reward))

        # Choose A' from S' using policy derived from Q
        (next_action, next_qual) = choose_action(estimator, next_state, epsilon)
        # Q(S, A) <- Q(S, A) + alpha(R + gamma * Q(S', A') - Q(S, A)
        # Set up the target value
        gamma = 0.9
        target = reward + gamma * (next_qual if not done else 0.)
        # Create a short lived training_data instance to manage the data we're learning
        data = training_data.training_data()
        data.add(state, next_action, target)

        # Augment data
        #data.augment()
        # Do the training
        train_input_fn = deep_model.numpy_train_fn(data.get_x(), data.get_y_digit(), data.get_reward())
        estimator.train(input_fn=train_input_fn)


        # Check for illegal moves
        if np.array_equal(state, next_state):
            print("Illegal move selected {}".format(illegal_count))
            total_illegals += 1
            illegal_count += 1
            if illegal_count > 10:
                print("No progress for 10 turns, breaking out")
                break
        else:
            illegal_count = 0

        # Update values for our tracking
        moves_taken += 1



        # S <- S'; A <- A'
        state = next_state
        action = next_action
        qual = next_qual

        # Exit if env says we're done
        if done:
            break

        print("")

    print("Took {} moves, Score: {}".format(moves_taken, total_reward))


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
