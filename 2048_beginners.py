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
       print(type(features))
       print(features)
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
    parser.add_argument('input', nargs='?', default='less_data')
    args = parser.parse_args()

    FILE_TRAIN = args.input
    FILE_TEST = args.input
    next_batch = my_input_fn(FILE_TRAIN)

    with tf.Session() as sess:
        first_batch = sess.run(next_batch)
    print(first_batch)

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

    # Train our model, use the previously function my_input_fn
    # Input to training is a file with training example
    # Stop training after 8 iterations of train data (epochs)
    classifier.train(
       input_fn=lambda: my_input_fn(FILE_TRAIN, True, 8))

    # Evaluate our model using the examples contained in FILE_TEST
    # Return value will contain evaluation_metrics such as: loss & average_loss
    evaluate_result = classifier.evaluate(
       input_fn=lambda: my_input_fn(FILE_TEST, False, 4))
    print("Evaluation results")
    for key in evaluate_result:
       print("   {}, was: {}".format(key, evaluate_result[key]))

    sys.exit(1)
    # Load data
    input_folder = args.input
    t = training_data.training_data()
    t.read(input_folder)
    if args.augment:
        t.augment()
    (x_training, x_dev) = t.get_x_partitioned()
    (y_training, y_dev) = t.get_y_partitioned()

    # Flatten boards out to 1 dimension
    number_of_training_items = x_training.shape[0]
    x_training = np.reshape(x_training, (number_of_training_items, 16)).astype(np.float32)
    print("{} training items".format(number_of_training_items))
    #y_training = y_training.astype(np.float32)

    # Create tensorflow data from loaded numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices({"x": x_training, "y": y_training})
    dataset = dataset.shuffle(2)
    dataset = dataset.batch(10)
    dataset = dataset.repeat(10)

    new_iterator = tf.data.Iterator.from_structure(dataset.output_types,
                                               dataset.output_shapes)
    next_element = new_iterator.get_next()
    training_init = new_iterator.make_initializer(dataset)

    # Create session
    sess = tf.InteractiveSession()

    # Print out data from dataset
    #sess.run(training_init)
    #while True:
    #    try:
    #        print(sess.run([next_element['x'], next_element['y']]))
    #    except tf.errors.OutOfRangeError:
    #        print("End of dataset")
    #        break

    # Model
    #x = tf.placeholder(tf.float32, [None, 16])
    with tf.name_scope('fc'):
        W = tf.Variable(tf.truncated_normal((16, 4), stddev=0.1))
        tf.summary.histogram('W', W)
        b = tf.Variable(tf.zeros([4]))
        tf.summary.histogram('b', b)
        #y_ = tf.placeholder(tf.float32, [None, 4])
        y_conv = tf.matmul(next_element['x'], W) + b

    # Better trainer from mnist_expert
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=next_element['y'], logits=y_conv))
    tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(next_element['y'], 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge summaries
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logdir', sess.graph)

    # Train model
    tf.global_variables_initializer().run()
    for i in range(50):
        sess.run(training_init)
        while True:
            try:
                (ts, my_y) = sess.run([train_step, y_conv])
            except tf.errors.OutOfRangeError:
                print("End of dataset")
                break
        sess.run(training_init)
        summary, ce, acc = sess.run([merged, cross_entropy, accuracy])
        print("Cross entropy {} and accuracy {}".format(ce, acc))
        writer.add_summary(summary, i)
    writer.close()
