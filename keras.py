#!/usr/bin/env python

"""Simple keras supervised learning model"""

from __future__ import print_function

import random
import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

import gym

import gym_2048
import training_data

def choose_action(model, observation, epsilon=0.1):
    """Choose best action from the esimator or random, based on epsilon
       Return both the action id and the estimated quality."""
    predictions = model.predict(np.reshape(observation, (-1, 16)))
    #print(predictions)
    if random.uniform(0, 1) > epsilon:
        chosen = np.argmax(predictions)
        #print("Choosing best action: {}".format(chosen))
    else:
        # Could change this to do based on probabilities
        chosen = random.randint(0, 3)
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

        # Update values for our tracking
        moves_taken += 1

        if moves_taken > 2000:
            break

        state = next_state

        # Exit if env says we're done
        if done:
            break

        #print("")
    return total_reward, moves_taken, total_illegals


def evaluate_model(model, epsilon):
  env = gym.make('2048-v0')
  env = env.unwrapped

  scores = []
  tt_reward = 0
  max_t_reward = 0.
  for i_episode in range(evaluation_episodes):
    print("Episode {}, using epsilon {}".format(i_episode, epsilon))
    (total_reward, moves_taken, total_illegals) = evaluate(model, env, epsilon, seed=456+i_episode, agent_seed=123+i_episode)
    print("Total reward {}, moves taken {} illegals {}".format(total_reward, moves_taken, total_illegals))
    scores.append({'total_reward': total_reward, 'highest': env.highest(), 'moves': moves_taken})
    tt_reward += total_reward
    max_t_reward = max(max_t_reward, total_reward)

  print("Average score: {}, Max score: {}".format(tt_reward / evaluation_episodes, max_t_reward))
  print(scores)
  env.close()

if __name__ == '__main__':
  print(tf.VERSION)
  print(tf.keras.__version__)
  
  inputs = 16
  outputs = 4
  filters = 256
  
  model = tf.keras.Sequential()
  # Seems like this wants flat input, fine, we'll reshape it
  model.add(layers.Reshape((4, 4, 1), input_shape=(inputs,)))
  
  conv_layers = 2
  for i in range(conv_layers):
    model.add(layers.Conv2D(filters=filters, kernel_size=(3, 3), padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
  
  model.add(layers.Conv2D(filters=1, kernel_size=(1, 1), padding='same'))
  model.add(layers.BatchNormalization())
  model.add(layers.Activation('relu'))

  # Output shape will be 16
  model.add(layers.Reshape((inputs,)))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(64, activation='relu'))
  model.add(layers.Dense(outputs, activation='softmax'))
  
  model.summary()
  
  model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
          loss='categorical_crossentropy',
          metrics=['accuracy'])
  
  td = training_data.training_data()
  td.import_csv(sys.argv[1])
  td.augment()
  td.normalize_boards()
  # Flatten board
  data = np.reshape(td.get_x(), (-1, 16))
  labels = td.get_y_one_hot()

  epsilon = 0.1
  evaluation_episodes = 10

  # Evaluate
  evaluate_model(model, epsilon)

  # Add tensorboard
  tensorboard = TensorBoard(log_dir='./logs',
    histogram_freq=0,
    write_graph=True,
    write_images=True)

  # Set early stopping
  early_stopping = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=2,
                              verbose=0, mode='auto')
  model.fit(data,
    labels,
    epochs=10,
    batch_size=32,
    validation_split=0.2,
    callbacks=[tensorboard, early_stopping])

  evaluate_model(model, epsilon)
