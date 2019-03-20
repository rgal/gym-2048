#!/usr/bin/env python

"""Simple keras supervised learning model"""

from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import TensorBoard, EarlyStopping

import training_data

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
