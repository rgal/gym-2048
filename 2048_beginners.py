import os

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, [None, 16])
W = tf.Variable(tf.zeros([16, 4]))
b = tf.Variable(tf.zeros([4]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 4])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# Load data
input_folder = 'augmented'
with open(os.path.join(input_folder, 'x.npy'), 'r') as f:
    x_training = np.load(f)
    number_of_items = x_training.shape[0]
    x_training = np.reshape(x_training, (number_of_items, 16))
with open(os.path.join(input_folder, 'y.npy'), 'r') as f:
    y_training = np.load(f)
for _ in range(1000):
  sess.run(train_step, feed_dict={x: x_training, y_: y_training})
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: x_training, y_: y_training}))
