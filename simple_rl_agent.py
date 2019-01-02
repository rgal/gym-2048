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

def choose_action(estimator, observation, epsilon=0.1):
    """Choose best action from the esimator or random, based on epsilon
       Return both the action id and the estimated quality."""
    if random.uniform(0, 1) > epsilon:
        prediction_2d = deep_model.get_predictions(estimator, observation.reshape((1, 4, 4)))
        predictions = prediction_2d.reshape((4))
        for i, v in enumerate(predictions):
            print("Action: {} Quality: {: .3f} {}".format(i, v, bar(v, -3, +3)))
        chosen = np.argmax(predictions)
        print("Choosing best action: {}".format(chosen))
    else:
        chosen = random.randint(0, 3)
        print("Choosing random action: {}".format(chosen))
    return chosen

def train(estimator, epsilon, replay_memory, seed=None, agent_seed=None):
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
    action = choose_action(estimator, state, epsilon)
    while 1:
        # Take action, observe R, S'
        next_state, reward, done, info = env.step(action)
        print(next_state.reshape(4, 4))
        total_reward += reward
        print("Score: {} Total reward: {}".format(env.score, total_reward))

        # Choose A' from S' using policy derived from Q
        next_action = choose_action(estimator, next_state, epsilon)

        # Add data to replay memory including immediate reward
        replay_memory.add(state, next_action, reward, next_state)

        # Augment data
        #data.augment()
        # Do the training
        minibatch_size = 32
        if replay_memory.size() >= minibatch_size:
            # Train from minibatch from replay memory
            sample_indexes = random.sample(range(replay_memory.size()), minibatch_size)
            sample_data = replay_memory.sample(sample_indexes)


            # Q(S, A) <- Q(S, A) + alpha(R + gamma * max(Q(S', A')) - Q(S, A)
            # Set up the target value as reward (from replay memory) + gamma * max()
            gamma = 0.9
            sample_rewards = sample_data.get_reward() # (batch_size, 1)
            sample_next_states = sample_data.get_next_x() # (batch_size, 4, 4)
            max_next_prediction = deep_model.get_maxq_per_state(estimator, sample_next_states)


            myu = 50.0
            sigma = 50.0
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
            # print("target")
            # print(target)
            # Target all at game score scale, normalize to help training
            target = (target - myu) / sigma
            # print("normalized target")
            # print(target)


            train_input_fn = deep_model.numpy_train_fn(sample_data.get_x(), sample_data.get_y_digit(), target)
            estimator.train(input_fn=train_input_fn)
        else:
            print("Not training, waiting for enough data {}".format(replay_memory.size()))


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

        # Exit if env says we're done
        if done:
            break

        print("")

    print("Took {} moves, Score: {}".format(moves_taken, total_reward))


    return total_reward, env.score, moves_taken, total_illegals

if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--epsilon', type=float, default=1, help="Probability of choosing random action instead of greedy")
    parser.add_argument('--epsilon-decay', type=float, default=0.9, help="How much to decay epsilon per episode")
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
    replay_memory = training_data.training_data(True)
    for i_episode in range(args.episodes):
        epsilon = max(args.epsilon * args.epsilon_decay ** i_episode, 0.1)
        print("Episode {}, using epsilon {}".format(i_episode, epsilon))
        (total_reward, score, moves_taken, illegal_count) = train(estimator, epsilon, replay_memory)
        scores.append({'score': score, 'total_reward': total_reward, 'moves': moves_taken, 'illegal_count': illegal_count, 'highest': env.highest(), 'epsilon': epsilon})
        #print(score)

    print(scores)
    with open('scores.csv', 'w') as f:
        w = csv.DictWriter(f, ['score', 'total_reward', 'moves', 'illegal_count', 'highest', 'epsilon'])
        w.writeheader()
        for s in scores:
            w.writerow(s)

    print("Average: {}".format(np.mean([s['score'] for s in scores])))
    # Close the environment
    env.close()

    end = datetime.datetime.now()
    taken = end - start
    print("{} episodes took {}. {:.1f} episodes per second".format(args.episodes, taken, args.episodes / taken.total_seconds()))
