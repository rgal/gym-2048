#!/usr/bin/env python

"""Simple keras supervised learning model"""

from __future__ import print_function

import csv
import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.metrics import sparse_top_k_categorical_accuracy

import gym

import gym_2048
import training_data

def choose_action(model, observation, epsilon=0.1):
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

def evaluate(model, env, epsilon, seed=None, agent_seed=None):
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
    return total_reward, moves_taken, total_illegals, info['highest']


def evaluate_model(model, epsilon, label='eval'):
  env = gym.make('2048-v0')
  env = env.unwrapped

  scores = []
  tt_reward = 0
  max_t_reward = 0.
  for i_episode in range(evaluation_episodes):
    (total_reward, moves_taken, total_illegals, highest) = evaluate(model, env, epsilon, seed=456+i_episode, agent_seed=123+i_episode)
    print("Episode {}, using epsilon {}, highest {}, total reward {}, moves taken {} illegals {}".format(i_episode, epsilon, highest, total_reward, moves_taken, total_illegals))
    scores.append({'total_reward': total_reward, 'highest': highest, 'moves': moves_taken, 'illegal_moves': total_illegals})
    tt_reward += total_reward
    max_t_reward = max(max_t_reward, total_reward)

  print("Average score: {}, Max score: {}".format(tt_reward / evaluation_episodes, max_t_reward))
  #print(scores)
  with open('scores_{}.csv'.format(label), 'w') as f:
    fieldnames = ['total_reward', 'highest', 'moves', 'illegal_moves']

    writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for s in scores:
      writer.writerow(s)

  env.close()

def top2_acc(labels, logits):
  return sparse_top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=2)

def top3_acc(labels, logits):
  return sparse_top_k_categorical_accuracy(y_true=labels, y_pred=logits, k=3)

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
  filters = 32
  residual_blocks = 4

  model = build_model(board_size, board_layers, outputs, filters, residual_blocks)

  # Summarise
  model.summary()

  model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
          loss='sparse_categorical_crossentropy',
          metrics=['accuracy', top2_acc, top3_acc])

  td = training_data.training_data()
  td.import_csv(sys.argv[1])
  #td.augment()
  x = td.get_x()
  print(td.size())
  unique_x, x_indices, x_inverse, x_counts = np.unique(x, axis=0, return_index=True, return_inverse=True, return_counts=True)
  print(unique_x)
  print(len(unique_x))
  print(x_indices)
  print(x_inverse)
  print(len(x_inverse))
  print(x_counts)
  print(np.amax(x_counts))
  #print(dict(zip(unique_x, x_counts)))
  for i in range(np.amax(x_counts) + 1):
    print("{} boards with {} copies".format(np.count_nonzero(x_counts ==i), i))
  sys.exit(0)

  td.shuffle()
  (training, validation) = td.split(0.8)
  training.augment()

  # Flatten board
  training_data = np.reshape(training.get_x_stacked().astype('float'), (-1, board_size * board_size * board_layers))
  training_labels = training.get_y_digit()
  validation_data = np.reshape(validation.get_x_stacked().astype('float'), (-1, board_size * board_size * board_layers))
  validation_labels = validation.get_y_digit()

  epsilon = 0.1
  evaluation_episodes = 10

  # Evaluate
  #evaluate_model(model, epsilon, 'pretraining')

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
  model.fit(training_data,
    training_labels,
    validation_data=(validation_data, validation_labels),
    epochs=10,
    batch_size=128,
    callbacks=[tensorboard, early_stopping])

  model.save('model.hdf5')

  # Report on training
  predictions = model.predict(validation_data)
  confusion = tf.math.confusion_matrix(validation_labels, np.argmax(predictions, axis=1))
  print("Confusion matrix (labels on left, predictions across the top)")
  print(confusion)

  evaluate_model(model, epsilon, 'trained_0_1')

