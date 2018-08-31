import os
import csv

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D

"""
Logisitic Regression using tensorflow using two features
Training data: ex1data2.csv
"""

data_directory = os.path.join(os.environ['VIRTUAL_ENV'], '..', 'data')
train_data_file = os.path.join(data_directory, 'ex1data2.csv')

with open(train_data_file, 'r') as reader:
    x1_data = []
    x2_data = []
    y_data = []
    for row in csv.reader(reader, delimiter=','):
        x1_data.append(float(row[0]))
        x2_data.append(float(row[1]))
        y_data.append(float(row[2]))

m = len(y_data)
num_features = 3
# Features
x0_data = np.ones((m, 1))
x1_data = np.reshape(x1_data, newshape=(m, 1))
x2_data = np.reshape(x2_data, newshape=(m, 1))
x_data = np.concatenate((x0_data, x1_data), axis=1)
x_data = np.concatenate((x_data, x2_data), axis=1)
# Value
y_data = np.reshape(y_data, newshape=(m, 1))

with tf.name_scope('Features'):
    x = tf.placeholder(tf.float32, shape=(m, num_features))
    y = tf.placeholder(tf.float32, shape=(m, 1))

with tf.name_scope('Weights'):
    theta = tf.Variable(tf.random_normal((num_features, 1)))
    bias = tf.Variable(tf.random_normal((1, 1)))

with tf.name_scope('Hypothesis'):
    z = tf.matmul(x, theta) + bias
    h_theta = tf.sigmoid(z)

with tf.name_scope('Cost'):
    j_theta = tf.reduce_sum(-y * tf.log(h_theta) - (1 - y) * tf.log(1 - h_theta))

with tf.name_scope('Optimization'):
    # optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=j_theta)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001).minimize(j_theta)

with tf.name_scope('Summaries'):
    tf.summary.scalar('Loss function', j_theta)
    merged_summaries = tf.summary.merge_all()

training_writer = tf.summary.FileWriter('/tmp/log-train', tf.get_default_graph())

training_steps = 15000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(0, training_steps+1):
        f_, summary, loss = sess.run([optimizer, merged_summaries, j_theta],
                                     feed_dict={x: x_data, y: y_data})
        if step % 1000 == 0:
            print('Step: {}, Loss: {}'.format(step, loss))
        training_writer.add_summary(summary, step)

    predicted_theta, predicted_bias = sess.run([theta, bias])
    predicted_h_theta = sess.run(h_theta, feed_dict={x: x_data})

# print(predicted_theta, predicted_bias)
# print(predicted_h_theta)

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x_data[:, 1], x_data[:, 2], y_data, c='r', marker='x')
# ax.set_xlabel('x2')
# ax.set_ylabel('x1')
# ax.set_zlabel('y')
# ax.plot(x2_data, x1_data, predicted_h_theta, marker='o', color='black', linestyle='dashed', linewidth=2)
#
# plt.show()

fig = plt.figure(1)
# ax = fig.gca(projection='3d')
ax = Axes3D(fig)
ax.scatter(x_data[:, 1], x_data[:, 2], y_data, color='r', marker='x', label='Training Data')
ax.scatter(x_data[:, 1], x_data[:, 2], predicted_h_theta, color='b', marker='o', label='Logistic Regression')
ax.set_xlabel('x1')
ax.set_ylabel('x2')
ax.set_zlabel('y/predicted')
ax.legend()
plt.show()
