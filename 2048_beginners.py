#!/usr/bin/env python

from __future__ import print_function

import os
import argparse

import tensorflow as tf
import numpy as np

import training_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', nargs='?', default='less_data')
    args = parser.parse_args()

    # Load data
    input_folder = args.input
    t = training_data.training_data()
    t.read(input_folder)
    x_training = t.get_x()
    y_training = t.get_y()

    # Flatten boards out to 1 dimension
    number_of_items = t.size()
    x_training = np.reshape(x_training, (number_of_items, 16)).astype(np.float32)
    y_training = y_training.astype(np.float32)

    # Create tensorflow data from loaded numpy arrays
    dataset = tf.data.Dataset.from_tensor_slices({"x": x_training, "y": y_training})
    dataset = dataset.shuffle(2)
    dataset = dataset.batch(1)
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
    W = tf.Variable(tf.truncated_normal((16, 4), stddev=0.1))
    b = tf.Variable(tf.zeros([4]))
    y = tf.nn.softmax(tf.matmul(next_element['x'], W) + b)
    #y_ = tf.placeholder(tf.float32, [None, 4])

    # Better trainer from mnist_expert
    y_conv = tf.matmul(next_element['x'], W) + b
    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=next_element['y'], logits=y_conv))
    tf.summary.scalar('cross_entropy', cross_entropy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    with tf.name_scope('accuracy'):
        with tf.name_scope('corrent_prediction'):
            correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(next_element['y'], 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)

    # Merge summaries
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('logdir', sess.graph)

    def print_stuff(training_init, W, b, y):
        sess.run(training_init)
        (thisW, thisb, thisy) = sess.run([W, b, y])
        print("Initial gradients: {}".format(thisW))
        print("Initial biases: {}".format(thisb))
        print("Initial output: {}".format(thisy))

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
