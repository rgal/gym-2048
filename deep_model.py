#/usr/bin/env python

import numpy as np
import tensorflow as tf

def get_maxq_per_state(estimator, states):
    # States is (batch_size, 4, 4)
    # Want to return (batch_size, 1) maximum Q
    qmax = np.amax(get_predictions(estimator, states), axis=1).reshape((-1, 1))
    return qmax

def get_predictions(estimator, states):
    """Get predictions for a number of states. States is (batch_size, 4, 4), returns numpy array of (batch_size, 4) with predictions for all actions."""
    predict_input_fn = numpy_predict_fn(states)
    prediction = estimator.predict(input_fn=predict_input_fn)
    list_predictions = [p['logits'] for p in prediction]
    np_array_prediction_values = np.array(list_predictions)
    return np_array_prediction_values

def numpy_predict_fn(observation):
   return tf.estimator.inputs.numpy_input_fn(x={'board': observation.reshape((-1,4,4,1)).astype(np.float32)},
       num_epochs=1,
       shuffle=False)

def numpy_train_fn(observation, action, reward):
   return tf.estimator.inputs.numpy_input_fn(x={'board': observation.reshape((-1,4,4,1)).astype(np.float32)},
       y={'action' : action.astype(np.int32), 'reward': reward},
       batch_size=1024,
       num_epochs=1,
       shuffle=False)

def my_input_fn(file_path, perform_shuffle=False, repeat_count=1, augment=False, batch_size=32):
   def decode_csv(line):
       parsed_line = tf.decode_csv(line, [[0] for i in range(17)] + [[0.0]])
       features = parsed_line[0:16]
       # Convert from list of tensors to one tensor
       features = tf.reshape(tf.cast(tf.stack(features), tf.float32), [4, 4, 1])
       action = parsed_line[16]
       reward = parsed_line[17]
       return {'board': features}, {'action': action, 'reward': reward}

   def hflip(feature, label):
       image = feature['board']
       flipped_image = tf.image.flip_left_right(image)
       #tf.Print(flipped_image, [image, flipped_image], "Image and flipped left right")
       new_action = tf.gather([0, 3, 2, 1], label['action'])
       #tf.Print(newlabel, [label['action'], newlabel], "Label and flipped left right")
       return {'board': flipped_image}, {'action': new_action, 'reward': label['reward']}

   def rotate_board(feature, label, k):
       image = feature['board']
       rotated_image = tf.image.rot90(image, 4 - k)
       #tf.Print(rotated_image, [image, rotated_image], "Image and rotated by k={}".format(k))
       new_action = label['action']
       new_action += k
       new_action %= 4
       #tf.Print(new_action, [label['action'], new_action], "Label and rotated by k={}".format(k))
       return {'board': rotated_image}, {'action': new_action, 'reward': label['reward']}

   def rotate90(feature, label):
       return rotate_board(feature, label, 1)

   def rotate180(feature, label):
       return rotate_board(feature, label, 2)

   def rotate270(feature, label):
       return rotate_board(feature, label, 3)

   dataset = (tf.data.TextLineDataset(file_path) # Read text file
       .skip(1) # Skip header row
       .map(decode_csv)) # Transform each elem by applying decode_csv fn

   if augment:
       parallel_map_calls = 4
       augmented = dataset.map(hflip, num_parallel_calls=parallel_map_calls)
       dataset = dataset.concatenate(augmented)
       r90 = dataset.map(rotate90, num_parallel_calls=parallel_map_calls)
       r180 = dataset.map(rotate180, num_parallel_calls=parallel_map_calls)
       r270 = dataset.map(rotate270, num_parallel_calls=parallel_map_calls)
       dataset = dataset.concatenate(r90)
       dataset = dataset.concatenate(r180)
       dataset = dataset.concatenate(r270)

   if perform_shuffle:
       # Randomizes input using a window of 256 elements (read into memory)
       dataset = dataset.shuffle(buffer_size=256)
   dataset = dataset.repeat(repeat_count) # Repeats dataset this # times
   dataset = dataset.batch(batch_size)  # Batch size to use
   iterator = dataset.make_one_shot_iterator()
   batch_features, batch_labels = iterator.get_next()
   return batch_features, batch_labels

def estimator(model_params):
    model_params['n_classes'] = 4
    return tf.estimator.Estimator(
        model_fn=my_model,
        model_dir='model_dir/{}_{}_{}_{}_{}'.format(model_params['dropout_rate'], model_params['residual_blocks'], model_params['filters'], '-'.join(map(str, model_params['fc_layers'])), '_bn' if model_params['batch_norm'] else ''), # Path to where checkpoints etc are stored
        params=model_params)

def residual_block(in_net, filters, dropout_rate, mode, bn=False):
    # Convolution layer 1
    # Input shape: [batch_size, 4, 4, 1]
    # Output shape: [batch_size, 4, 4, 16]
    net = tf.layers.conv2d(
      inputs=in_net,
      filters=filters,
      kernel_size=[3, 3],
      padding="same",
      activation=None)

    if bn:
        # Batch norm
        net = tf.layers.batch_normalization(
            inputs=net,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

    # Non linearity
    net = tf.nn.relu(net)

    # Convolution layer 1
    # Input shape: [batch_size, 4, 4, 1]
    # Output shape: [batch_size, 4, 4, 16]
    net = tf.layers.conv2d(
      inputs=net,
      filters=filters,
      kernel_size=[3, 3],
      padding="same",
      activation=None)

    if bn:
        # Batch norm
        net = tf.layers.batch_normalization(
            inputs=net,
            training=mode == tf.estimator.ModeKeys.TRAIN
        )

    # Non linearity
    net = tf.nn.relu(net)

    # Add skip connection
    return in_net + net

def log2(x):
    """Log to base 2"""
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(2, dtype=numerator.dtype))
    return numerator / denominator

def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""

    l0 = features['board']

    # Batch normalize to get distribution closer to zero
    l0_bn = tf.layers.batch_normalization(
        inputs=l0,
        training=mode == tf.estimator.ModeKeys.TRAIN
    )

    # Convolution layer 1
    # Input shape: [batch_size, 4, 4, 1]
    # Output shape: [batch_size, 4, 4, 16]
    block_inout = tf.layers.conv2d(
      inputs=l0_bn,
      filters=params['filters'],
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    #for filters in params['conv_layers']:
    for res_block in range(params['residual_blocks']):
        block_inout = residual_block(block_inout, params['filters'], params['dropout_rate'], mode, params['batch_norm'])

    # Flatten into a batch of vectors
    # Input shape: [batch_size, 4, 4, 16]
    # Output shape: [batch_size, 4 * 4 * 16]
    net = tf.reshape(block_inout, [-1, 4 * 4 * params['filters']])

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
    #predicted_classes = tf.argmax(logits, 1)
    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {
            #'class_ids': predicted_classes[:, tf.newaxis],
            #'probabilities': tf.nn.softmax(logits),
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # Compute loss on actions compared to records
    action_labels = labels['action'] # Shape [batch_size, 1]
    # Then calculate loss with softmax cross entropy
    action_loss = tf.losses.sparse_softmax_cross_entropy(labels=action_labels, logits=logits)

    # Compute loss based no rewards compared to those from environment
    reward_labels = labels['reward'] # Shape [batch_size, 1]

    # Only have reward from environment from one action so gather predicted
    # rewards for those actions and calculate MSE compared to environment
    batch_size = tf.shape(labels['action'])[0]
    action_idx = tf.reshape(tf.range(batch_size),[-1, 1])
    action_labels_reshaped = tf.reshape(labels['action'],[-1, 1])
    action_reshaped = tf.reshape(tf.concat([action_idx, action_labels_reshaped],axis=1), [-1, 1, 2])
    gathered_logits = tf.gather_nd(logits, action_reshaped)
    reward_loss = tf.losses.mean_squared_error(tf.reshape(reward_labels, [-1, 1]), gathered_logits)

    # Batch normalize to get distribution closer to zero
#    reward_bn = tf.layers.batch_normalization(
#        inputs=tf.reshape(labels['reward'], [-1, 1]),
#        training=mode == tf.estimator.ModeKeys.TRAIN
#    )

    # Calculate Q loss on the Q of the input action
    #q_loss = tf.losses.mean_squared_error(labels['reward'], tf.reshape(logits, [-1, 1]))

    # Select loss (action or reward)
    loss = reward_loss

    # Compute evaluation metrics.
#    accuracy = tf.metrics.accuracy(labels=action_labels,
#                                   predictions=predicted_classes,
#                                   name='acc_op')
#    metrics = {'accuracy': accuracy}
#    tf.summary.scalar('accuracy', accuracy[1])

    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(
            mode, loss=loss)

    # Create training op.
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdamOptimizer(params.get('learning_rate', 0.05))

    # Add extra dependencies for batch normalisation
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


