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

def my_input_fn(file_path, perform_shuffle=False, repeat_count=1, augment=False):
   def decode_csv(line):
       parsed_line = tf.decode_csv(line, [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0]])
       features = parsed_line[0:16]
       label = parsed_line[16]
       d = dict(zip(feature_names, features)), label
       return d

   def augment_data(dataset):
       raise NotImplementedError

   dataset = (tf.data.TextLineDataset(file_path) # Read text file
       #.skip(1) # Skip header row
       .map(decode_csv)) # Transform each elem by applying decode_csv fn
   if augment:
       dataset = augment_data(dataset)
   if perform_shuffle:
       # Randomizes input using a window of 256 elements (read into memory)
       dataset = dataset.shuffle(buffer_size=256)
   dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
   dataset = dataset.batch(32)  # Batch size to use
   iterator = dataset.make_one_shot_iterator()
   batch_features, batch_labels = iterator.get_next()
   return batch_features, batch_labels

def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""

    # Input layer
    fcs = tf.feature_column.input_layer(features, params['feature_columns'])

    # Re-shape from flat features to 4d vector
    # Input shape: [batch_size, 16]
    # Output shape: [batch_size, 4, 4, 1]
    net = tf.reshape(fcs, [-1, 4, 4, 1])

    for filters in params['conv_layers']:
        # Convolution layer 1
        # Input shape: [batch_size, 4, 4, 1]
        # Output shape: [batch_size, 4, 4, 16]
        net = tf.layers.conv2d(
          inputs=net,
          filters=filters,
          kernel_size=[3, 3],
          padding="same",
          activation=tf.nn.relu)

        # Add dropout operation
        net = tf.layers.dropout(
            inputs=net, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN)

    # Flatten into a batch of vectors
    # Input shape: [batch_size, 4, 4, 16]
    # Output shape: [batch_size, 4 * 4 * 16]
    net = tf.reshape(net, [-1, 4 * 4 * params['conv_layers'][-1]])

    for units in params['fc_layers']:
        # Fully connected layer
        # Input shape: [batch_size, 4 * 4 * 16]
        # Output shape: [batch_size, 16]
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

        # Add dropout operation
        net = tf.layers.dropout(
            inputs=net, rate=params['dropout_rate'], training=mode == tf.estimator.ModeKeys.TRAIN)

    # Compute logits (1 per class).
    logits = tf.layers.dense(net, params['n_classes'], activation=None)

    # Compute predictions.
    predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],
            'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss.
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # Compute evaluation metrics.
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss, eval_metric_ops=metrics)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(params.get('learning_rate', 0.05))
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--augment', default=False, action='store_true', help='augment data')
    parser.add_argument('-e', '--epochs', type=int, default=5, help='How many times to go through data')
    parser.add_argument('train_input', nargs='?', default='train.csv')
    parser.add_argument('test_input', nargs='?', default='test.csv')
    args = parser.parse_args()

    FILE_TRAIN = args.train_input
    FILE_TEST = args.test_input

    for learning_rate in [0.1, 0.05]:
      for dropout_rate in [0.25, 0.5, 0.75]:
        print("Learning rate: {}, dropout rate: {}".format(learning_rate, dropout_rate))

        # Create the feature_columns, which specifies the input to our model.
        # All our input features are numeric, so use numeric_column for each one.
        feature_columns = [tf.feature_column.numeric_column(k) for k in feature_names]

        # Create a deep neural network regression classifier.
        # Build custom classifier
        classifier = tf.estimator.Estimator(
            model_fn=my_model,
            model_dir='model_dir/{}_{}'.format(learning_rate, dropout_rate), # Path to where checkpoints etc are stored
            params={
                'feature_columns': feature_columns,
                'n_classes': 4,
                'dropout_rate': dropout_rate,
                'learning_rate': learning_rate,
                'conv_layers': [32, 32],
                'fc_layers': [64, 16],
            })

        for epoch in range(args.epochs):
            # Train our model, use the previously function my_input_fn
            # Input to training is a file with training example
            classifier.train(
               input_fn=lambda: my_input_fn(FILE_TRAIN, True))

            if epoch % 10 == 0:
                print("Epoch: {}".format(epoch))
                # Evaluate our model
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
