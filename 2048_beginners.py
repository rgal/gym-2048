#!/usr/bin/env python

from __future__ import print_function

import os
import argparse
import sys

import tensorflow as tf
import numpy as np

import training_data

feature_names = [
'00', '10', '20', '30',
'01', '11', '21', '31',
'02', '12', '22', '32',
'03', '13', '23', '33',
]

def my_input_fn(file_path, perform_shuffle=False, repeat_count=1):
   def decode_csv(line):
       parsed_line = tf.decode_csv(line, [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
       features = parsed_line[0:16]
       label = parsed_line[16]
       d = dict(zip(feature_names, features)), label
       return d

   dataset = (tf.data.TextLineDataset(file_path) # Read text file
       #.skip(1) # Skip header row
       .map(decode_csv)) # Transform each elem by applying decode_csv fn
   if perform_shuffle:
       # Randomizes input using a window of 256 elements (read into memory)
       dataset = dataset.shuffle(buffer_size=256)
   dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
   dataset = dataset.batch(32)  # Batch size to use
   iterator = dataset.make_one_shot_iterator()
   batch_features, batch_labels = iterator.get_next()
   return batch_features, batch_labels

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--augment', default=False, action='store_true', help='augment data')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='How many times to go through data')
    parser.add_argument('input', nargs='?', default='less_data')
    args = parser.parse_args()

    FILE_TRAIN = 'train.csv'
    FILE_TEST = 'test.csv'

    # Print out a batch of data
    # next_batch = my_input_fn(FILE_TRAIN)
    # with tf.Session() as sess:
    #     first_batch = sess.run(next_batch)
    # print(first_batch)

    # Create the feature_columns, which specifies the input to our model.
    # All our input features are numeric, so use numeric_column for each one.
    feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

    # Create a deep neural network regression classifier.
    # Use the DNNClassifier pre-made estimator
    classifier = tf.estimator.DNNClassifier(
       feature_columns=feature_columns, # The input features to our model
       hidden_units=[16, 16], # Two layers, each with 10 neurons
       n_classes=4,
       model_dir='model_dir') # Path to where checkpoints etc are stored

    for epoch in range(args.epochs):
        # Train our model, use the previously function my_input_fn
        # Input to training is a file with training example
        # Stop training after 8 iterations of train data (epochs)
        classifier.train(
           input_fn=lambda: my_input_fn(FILE_TRAIN, True))

        if epoch % 10 == 0:
            # Evaluate our model using the examples contained in FILE_TEST
            # Return value will contain evaluation_metrics such as: loss & average_loss
            evaluate_result = classifier.evaluate(
               input_fn=lambda: my_input_fn(FILE_TRAIN, False, 4), name='train')
            print("Evaluation results")
            for key in evaluate_result:
               print("   {}, was: {}".format(key, evaluate_result[key]))
            evaluate_result = classifier.evaluate(
               input_fn=lambda: my_input_fn(FILE_TEST, False, 4), name='test')
            print("Evaluation results")
            for key in evaluate_result:
               print("   {}, was: {}".format(key, evaluate_result[key]))
