#!/usr/bin/env python

"""Simple keras supervised learning model"""

from __future__ import print_function

import csv
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping, LearningRateScheduler
from tensorflow.keras.metrics import sparse_top_k_categorical_accuracy

import gym

import gym_2048
import training_data

def choose_action(model, observation, epsilon=0.):
    """Choose best action from the esimator or random, based on epsilon
       Return both the action id and the estimated quality."""
    predictions = np.reshape(model.predict(np.reshape(observation.astype('float32'), (-1, 256))), (4, ))
    #print(predictions)
    if random.uniform(0, 1) > epsilon:
        chosen = np.argmax(predictions)
        #print("Choosing best action: {}".format(chosen))
    else:
        # Choose random action weighted by predictions
        chosen = np.random.choice(4, 1, p=predictions)
        #print("Choosing random action: {}".format(chosen))
    return chosen

def evaluate_episode(model, env, epsilon, seed=None, agent_seed=None):
    """Evaluate estimator for one episode.
    seed (optional) specifies the seed for the game.
    agent_seed specifies the seed for the agent."""
    #print("Evaluating")
    # Initialise seed for environment
    if seed:
        env.seed(seed)
    else:
        env.seed()
    # Initialise seed for agent choosing epsilon greedy
    if agent_seed:
        random.seed(agent_seed)
    else:
        random.seed()

    total_reward = 0.0
    total_illegals = 0
    moves_taken = 0

    # Initialise S
    state = env.reset()
    # Choose A from S using policy derived from Q
    while 1:
        #print(state.reshape(4, 4))
        action = choose_action(model, state, epsilon)

        # Take action, observe R, S'
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        #print("Score: {} Total reward: {}".format(env.score, total_reward))

        # Count illegal moves
        if info['illegal_move']:
            total_illegals += 1

        # Update values for our tracking
        moves_taken += 1

        if moves_taken > 2000:
            break

        state = next_state

        # Exit if env says we're done
        if done:
            break

        #print("")
    return total_reward, moves_taken, total_illegals, int(info['highest'])


def evaluate_model(model, episodes, epsilon):
    env = gym.make('2048-v0')
    env = env.unwrapped

    scores = []
    for i_episode in range(episodes):
        (total_reward, moves_taken, total_illegals, highest) = evaluate_episode(model, env, epsilon, seed=456+i_episode, agent_seed=123+i_episode)
        print("Episode {}, using epsilon {}, highest {}, total reward {}, moves taken {} illegals {}".format(i_episode, epsilon, highest, total_reward, moves_taken, total_illegals))
        scores.append({'total_reward': total_reward, 'highest': highest, 'moves': moves_taken, 'illegal_moves': total_illegals})

    env.close()

    total_score = sum([s['total_reward'] for s in scores])
    average_score = total_score / episodes
    max_score = max(s['total_reward'] for s in scores)
    highest_tile = max(s['highest'] for s in scores)

    print("Highest tile: {}, Average score: {}, Max score: {}".format(highest_tile, average_score, max_score))

    return {
        "Average score": average_score,
        "Max score": max_score,
        "Highest tile": highest_tile,
        "Episodes" : scores,
    }

def report_evaluation_results(results, label='eval'):
    scores = results['Episodes']
    with open('scores_{}.csv'.format(label), 'w') as f:
        fieldnames = ['total_reward', 'highest', 'moves', 'illegal_moves']

        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for s in scores:
            writer.writerow(s)

def build_model(board_size=4, board_layers=16, outputs=4, filters=64, residual_blocks=4):
    # Functional API model
    inputs = layers.Input(shape=(board_size * board_size * board_layers,))
    x = layers.Reshape((board_size, board_size, board_layers))(inputs)

    # Initial convolutional block
    x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # residual blocks
    for i in range(residual_blocks):
        # x at the start of a block
        temp_x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(x)
        temp_x = layers.BatchNormalization()(temp_x)
        temp_x = layers.Activation('relu')(temp_x)
        temp_x = layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same')(temp_x)
        temp_x = layers.BatchNormalization()(temp_x)
        x = layers.add([x, temp_x])
        x = layers.Activation('relu')(x)

    # policy head
    x = layers.Conv2D(filters=2, kernel_size=(1, 1), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Flatten()(x)
    predictions = layers.Dense(outputs, activation='softmax')(x)

    # Create model
    return models.Model(inputs=inputs, outputs=predictions)

if __name__ == '__main__':
    print("Tensorflow version: {}".format(tf.__version__))
    print("Tensorflow keras version: {}".format(tf.keras.__version__))

    board_size = 4
    board_squares = board_size * board_size
    board_layers = 16 # Layers of game board to represent different numbers
    outputs = 4
    filters = 64
    residual_blocks = 8

    model = build_model(board_size, board_layers, outputs, filters, residual_blocks)

    # Summarise
    model.summary()

    td = training_data.training_data()
    td.import_csv(sys.argv[1])
    td.shuffle()
    (training, validation) = td.split(0.8)
    training.augment()
    training.make_boards_unique()

    # Flatten board
    training_data = np.reshape(training.get_x_stacked().astype('float'), (-1, board_size * board_size * board_layers))
    training_labels = training.get_y_digit()
    validation_data = np.reshape(validation.get_x_stacked().astype('float'), (-1, board_size * board_size * board_layers))
    validation_labels = validation.get_y_digit()

    epsilon = 0.1
    evaluation_episodes = 10

    # Evaluate
    results = evaluate_model(model, evaluation_episodes, epsilon)
    report_evaluation_results(results, 'pretraining')

    # Add tensorboard
    tensorboard = TensorBoard(log_dir='./logs',
                              histogram_freq=0,
                              write_graph=True,
                              write_images=True)

    # Set early stopping
    early_stopping = EarlyStopping(monitor='val_loss',
                                   min_delta=0,
                                   patience=3,
                                   verbose=0, mode='auto')

    def scheduler(epoch):
        initial_epochs = 10
        decay_rate = 0.05
        if epoch < initial_epochs:
            return 0.001
        else:
            return 0.001 * tf.math.exp(decay_rate * (initial_epochs - epoch))

    lr_callback = LearningRateScheduler(scheduler, verbose=1)

    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(training_data,
              training_labels,
              validation_data=(validation_data, validation_labels),
              epochs=5,
              batch_size=128,
              callbacks=[tensorboard,
                         early_stopping,
                         lr_callback])

    model.save('model.hdf5')

    # Report on training
    predictions = model.predict(validation_data)
    confusion = tf.math.confusion_matrix(validation_labels, np.argmax(predictions, axis=1))
    print("Confusion matrix (labels on left, predictions across the top)")
    print(confusion)

    results = evaluate_model(model, evaluation_episodes, epsilon)
    report_evaluation_results(results, 'trained_0_1')
