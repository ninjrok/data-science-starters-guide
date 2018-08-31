import os
import csv

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

"""
Linear Regression using tensorflow using a single feature
Training data: ex1data1.csv
"""

data_directory = os.path.join(os.environ['VIRTUAL_ENV'], '..', 'data')
train_data_file = os.path.join(data_directory, 'ex1data1.csv')

with open(train_data_file, 'r') as reader:
    x_data = []
    y_data = []
    for row in csv.reader(reader, delimiter=','):
        x_data.append(float(row[0]))
        y_data.append(float(row[1]))

m = len(y_data)
x_data = np.reshape(x_data, newshape=(m, 1))
y_data = np.reshape(y_data, newshape=(m, 1))

with tf.name_scope('Placeholders'):
    x = tf.placeholder(tf.float32, shape=(m, 1))
    y = tf.placeholder(tf.float32, shape=(m, 1))

with tf.name_scope('Weights'):
    theta0 = tf.Variable(tf.random_normal((1, 1)))
    theta1 = tf.Variable(tf.random_normal((1, 1)))

with tf.name_scope('Hypothesis'):
    h_theta = tf.add(tf.matmul(x, theta1), theta0)

with tf.name_scope('Cost'):
    j_theta = tf.reduce_sum((y - h_theta) ** 2)

with tf.name_scope('Optimization'):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(loss=j_theta)

with tf.name_scope('Summaries'):
    tf.summary.scalar('Loss function', j_theta)
    merged_summaries = tf.summary.merge_all()

training_writer = tf.summary.FileWriter('/tmp/lr-train', tf.get_default_graph())

training_steps = 15000
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for step in range(0, training_steps):
        f_, summary, loss = sess.run([optimizer, merged_summaries, j_theta], feed_dict={x: x_data, y: y_data})
        if step % 1000 == 0:
            print('Step: {}, Loss: {}'.format(step, loss))
        training_writer.add_summary(summary, step)

    predicted_theta1, predicted_theta0 = sess.run([theta1, theta0])
    predicted_h_theta = sess.run(h_theta, feed_dict={x: x_data})

print('Predicted theta values are: {}\n{}'.format(predicted_theta0, predicted_theta1))

r2 = pearsonr(y_data, predicted_h_theta)[0] ** 2
rms = np.sqrt(mean_squared_error(y_data, predicted_h_theta))

# fig, ax = plt.subplots()
# plt.title('True values vs Prediction. R2: {}, RMS: {}'.format(r2, rms))
# ax.scatter(x_data, y_data, c='r', marker='x', label='True value')
# ax.scatter(x_data, predicted_h_theta, c='b', marker='o', label='Prediction')
# plt.xlabel('Population of city in 10,000s')
# plt.ylabel('Profit in $10,000s')
# ax.legend()
# plt.show()
# plt.close()

fig, ax = plt.subplots()
plt.title('Linear regression. Theta1={}, Theta0={}'.format(predicted_theta1, predicted_theta0))
ax.scatter(x_data, y_data, c='r', marker='x', label='Train data')
ax.plot(x_data, predicted_h_theta, 'k--', lw=1.5, label='Prediction')
plt.xlabel('Population of city in 10,000s')
plt.ylabel('Profit in $10,000s')
ax.legend()
plt.show()

plt.close()
reader.close()
sess.close()
