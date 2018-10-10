from __future__ import print_function

import argparse
import random
import datetime
import numpy as np

import gym

import gym_2048

import training_data
import deep_model

def choose_action(estimator, observation, epsilon=0.9):
    """Choose best action from the esimator or random, based on epsilon"""
    print(observation.reshape((4, 4)))
    if (random.uniform(0, 1) < epsilon):
        predict_input_fn = deep_model.numpy_predict_fn(observation)
        prediction = list(estimator.predict(input_fn=predict_input_fn))[0]
        print(prediction['logits'])
        chosen = np.argmax(prediction['logits'])
        print("Choosing best action: {}".format(chosen))
    else:
        chosen = random.randint(0, 3)
        print("Choosing random action: {}".format(chosen))
    return chosen

def train(estimator, seed=None, agent_seed=None):
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

    # Initialise S, A
    action = choose_action(estimator, observation)

    illegal_count = 0
    total_reward = 0.0
    moves_taken = 0
    while 1:
        #env.render()
        #print(observation)
        #print "Action: {}".format(action)
        #last_observation = tuple(observation)
        #print "Observation: {}".format(last_observation)
        # Take action, observe R, S'
        next_observation, reward, done, info = env.step(action)
        total_reward += reward
        print("Score: {}".format(total_reward))
        #print "New Observation: {}, reward: {}, done: {}, info: {}".format(next_observation, reward, done, info)
        # Record what we did in a particular state
        if np.array_equal(observation, next_observation):
            print("Illegal move selected")
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

        # Choose A' from S' using policy derived from Q
        next_action = choose_action(estimator, observation)

        observation = next_observation
        action = next_action
        print("")

    print("Took {} moves, Score: {}".format(moves_taken, total_reward))
    scaled_reward = total_reward / 10000 - 1.0
    print("Scaled reward: {}".format(scaled_reward))
    # Apply training to model using history of boards, actions and total reward
    rreward = np.full((moves_taken, 1), scaled_reward, dtype=np.float)
    train_input_fn = deep_model.numpy_train_fn(data.get_x(), data.get_y_digit(), rreward)
    estimator.train(input_fn=train_input_fn)
    return total_reward

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episodes', type=int, default=1)
    args = parser.parse_args()

    env = gym.make('2048-v0')

    # Load estimator
    params = {
        'learning_rate': 0.01,
        'dropout_rate': 0,
        'residual_blocks': 2,
        'filters': 16,
        'batch_norm': True,
        'fc_layers': [512, 512],
    }
    estimator = deep_model.estimator(params)

    start = datetime.datetime.now()
    scores = list()
    for i_episode in range(args.episodes):
        print("Episode {}".format(i_episode))
        score = train(estimator)
        scores.append(score)
        print(score)

    print(scores)
    # Close the environment
    env.close()

    end = datetime.datetime.now()
    taken = end - start
    print("{} episodes took {}. {:.1f} episodes per second".format(args.episodes, taken, args.episodes / taken.total_seconds()))
