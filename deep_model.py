#/usr/bin/env python

import tensorflow as tf

def my_input_fn(file_path, perform_shuffle=False, repeat_count=1, augment=False, batch_size=32):
   def decode_csv(line):
       parsed_line = tf.decode_csv(line, [[0] for i in range(17)])
       features = parsed_line[0:16]
       # Convert from list of tensors to one tensor
       features = tf.reshape(tf.cast(tf.stack(features), tf.float32), [4, 4, 1])
       label = parsed_line[16]
       return {'board': features}, label

   def hflip(feature, label):
       image = feature['board']
       flipped_image = tf.image.flip_left_right(image)
       #tf.Print(flipped_image, [image, flipped_image], "Image and flipped left right")
       newlabel = tf.gather([0, 3, 2, 1], label)
       #tf.Print(newlabel, [label, newlabel], "Label and flipped left right")
       return {'board': flipped_image}, newlabel

   def rotate_board(feature, label, k):
       image = feature['board']
       rotated_image = tf.image.rot90(image, 4 - k)
       #tf.Print(rotated_image, [image, rotated_image], "Image and rotated by k={}".format(k))
       newlabel = label
       newlabel += k
       newlabel %= 4
       #tf.Print(newlabel, [label, newlabel], "Label and rotated by k={}".format(k))
       return {'board': rotated_image}, newlabel

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
       augmented = dataset.map(hflip, num_parallel_calls=1)
       r90 = dataset.map(rotate90, num_parallel_calls=4)
       r180 = dataset.map(rotate180, num_parallel_calls=4)
       r270 = dataset.map(rotate270, num_parallel_calls=4)
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

def my_model(features, labels, mode, params):
    """DNN with three hidden layers, and dropout of 0.1 probability."""

    l0 = features['board']

    # Convolution layer 1
    # Input shape: [batch_size, 4, 4, 1]
    # Output shape: [batch_size, 4, 4, 16]
    block_inout = tf.layers.conv2d(
      inputs=l0,
      filters=params['filters'],
      kernel_size=[3, 3],
      padding="same",
      activation=tf.nn.relu)

    #for filters in params['conv_layers']:
    for res_block in range(params['residual_blocks']):
        block_inout = residual_block(block_inout, params['filters'], params['dropout_rate'], mode, params['bn'])

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

    # Add extra dependencies for batch normalisation
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


