#!/usr/bin/env python

from __future__ import print_function

import argparse
import csv
import random

import tensorflow as tf

import deep_model

def evaluate_model(training_file, test_file, epochs, augment, batch_size, model_params):
    print("Learning rate: {}, dropout rate: {}, {} residual blocks, {} filters, bn: {}, fc: {}, augmenting {}, batch_size: {}".format(model_params['learning_rate'], model_params['dropout_rate'], model_params['residual_blocks'], model_params['filters'], model_params['batch_norm'], model_params['fc_layers'], augment, batch_size))

    # Create a deep neural network regression classifier.
    # Build custom classifier
    classifier = deep_model.estimator(model_params)

    results = {}
    for epoch in range(epochs):
        # Train our model, use the previously function my_input_fn
        # Input to training is a file with training example
        classifier.train(
           input_fn=lambda: deep_model.my_input_fn(training_file, True, 1, augment, batch_size))

        if epoch % 10 == 0 or epoch == (epochs - 1):
            print("Epoch: {}".format(epoch))
            # Evaluate our model
            # Return value will contain evaluation_metrics such as: loss & average_loss
            # Evaluate on training set
            evaluate_result = classifier.evaluate(
               input_fn=lambda: deep_model.my_input_fn(training_file, False, 4, False, batch_size), name='train')
            print("Evaluation results")
            for key in evaluate_result:
               print("   {}, was: {}".format(key, evaluate_result[key]))
            results['training_loss'] = evaluate_result['loss']
            results['training_accuracy'] = evaluate_result['accuracy']
            # Evaluate on test set
            evaluate_result = classifier.evaluate(
               input_fn=lambda: deep_model.my_input_fn(test_file, False, 4, False, batch_size), name='test')
            print("Evaluation results")
            for key in evaluate_result:
               print("   {}, was: {}".format(key, evaluate_result[key]))
            results['test_loss'] = evaluate_result['loss']
            results['test_accuracy'] = evaluate_result['accuracy']
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', type=int, default=5, help='How many times to go through data')
    parser.add_argument('-p', '--param-sets', type=int, default=1, help='How many different sets of hyperparameters to try')
    parser.add_argument('train_input', nargs='?', default='train.csv')
    parser.add_argument('test_input', nargs='?', default='test.csv')
    args = parser.parse_args()

    batch_norm = True
    augment = True
    dropout_rate = 0#random.random() * 0.5
    batch_size = 32#2 ** random.randint(5, 9)
    with open('hp_search.csv', 'w') as csvfile:
        fieldnames = ['run',
            'learning_rate',
            'dropout_rate',
            'residual_blocks',
            'filters',
            'augment',
            'batch_norm',
            'fc1',
            'fc2',
            'batch_size',
            'training_loss',
            'training_accuracy',
            'test_loss',
            'test_accuracy',
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for m in range(args.param_sets):
            print("Evaluating parameter set: ({}/{})".format(m + 1, args.param_sets))
            params = {
                'learning_rate': 10 ** (random.random() * -4.0),
                'dropout_rate': 0,
                'residual_blocks': random.randint(1, 10),
                'filters': 2 ** random.randint(2, 6),
                'batch_norm': True,
                'fc_layers': [2 ** random.randint(6, 9), 2 ** random.randint(4, 9)],
            }
            results = evaluate_model(args.train_input, args.test_input, args.epochs, augment, batch_size, params)
            results['run'] = m
            results['learning_rate'] = params['learning_rate']
            results['dropout_rate'] = params['dropout_rate']
            results['residual_blocks'] = params['residual_blocks']
            results['filters'] = params['filters']
            results['batch_norm'] = params['batch_norm']
            results['fc1'] = params['fc_layers'][0]
            results['fc2'] = params['fc_layers'][1]
            results['augment'] = augment
            results['batch_size'] = batch_size
            writer.writerow(results)
