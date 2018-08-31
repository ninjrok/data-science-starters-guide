import os

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.examples.tutorials.mnist import input_data


def conv2d(x, kernel, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, kernel, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2, dropout_prob=0.5):
    return tf.nn.dropout(tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1], padding='SAME'),
                         keep_prob=dropout_prob)


def plot_data(x, y1, title, xlabel, ylabel, xplabel, yplabel, y2=None):
    if not y2:
        y2 = y1
        y1 = x
    plt.plot(range(len(x)), y1, 'b', label=xplabel)
    plt.plot(range(len(x)), y2, 'r', label=yplabel)
    plt.title(title)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.figure()
    plt.show()


venv_dir = os.environ['VIRTUAL_ENV']
data_dir = os.path.join(venv_dir, 'data', 'fashion')

data = input_data.read_data_sets(data_dir, source_url='http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/',
                                 one_hot=True)
label_dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot',
}

plt.figure(figsize=[5, 5])

# Display the first image in training data
plt.subplot(121)
curr_img = np.reshape(data.train.images[0], (28, 28))
curr_lbl = np.argmax(data.train.labels[0, :])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")

# Display the first image in testing data
plt.subplot(122)
curr_img = np.reshape(data.test.images[0], (28, 28))
curr_lbl = np.argmax(data.test.labels[0, :])
plt.imshow(curr_img, cmap='gray')
plt.title("(Label: " + str(label_dict[curr_lbl]) + ")")
plt.show()

# Reshape training and testing image
train_X = data.train.images.reshape(-1, 28, 28, 1)
test_X = data.test.images.reshape(-1, 28, 28, 1)

train_y = data.train.labels
test_y = data.test.labels

training_iters = 35
learning_rate = 0.001
batch_size = 128
keep_prob = 0.5

# MNIST total classes (0-9 digits)
n_classes = 10

with tf.name_scope('placeholders'):
    x = tf.placeholder(tf.float32, (None, 28, 28, 1))
    y = tf.placeholder(tf.float32, (None, 10))

with tf.name_scope('conv-layer-1'):
    kernel1 = tf.get_variable('K1', shape=(3, 3, 1, 32), initializer=tf.contrib.layers.xavier_initializer())
    bias1 = tf.get_variable('B1', shape=(32), initializer=tf.contrib.layers.xavier_initializer())
    conv1 = conv2d(x, kernel1, bias1)
    conv1 = maxpool2d(conv1, k=2, dropout_prob=keep_prob)

with tf.name_scope('conv-layer-2'):
    kernel2 = tf.get_variable('K2', shape=(3, 3, 32, 64), initializer=tf.contrib.layers.xavier_initializer())
    bias2 = tf.get_variable('B2', shape=(64), initializer=tf.contrib.layers.xavier_initializer())
    conv2 = conv2d(conv1, kernel2, bias2)
    conv2 = maxpool2d(conv2, k=2, dropout_prob=keep_prob)

with tf.name_scope('conv-layer-3'):
    kernel3 = tf.get_variable('K3', shape=(3, 3, 64, 128), initializer=tf.contrib.layers.xavier_initializer())
    bias3 = tf.get_variable('B3', shape=(128), initializer=tf.contrib.layers.xavier_initializer())
    conv3 = conv2d(conv2, kernel3, bias3)
    conv3 = maxpool2d(conv3, k=2, dropout_prob=keep_prob)

with tf.name_scope('fully-connected-layer'):
    w = tf.get_variable('W', shape=(4 * 4 * 128, 128), initializer=tf.contrib.layers.xavier_initializer())
    bias = tf.get_variable('B', shape=(128), initializer=tf.contrib.layers.xavier_initializer())
    # Flatten
    fc = tf.reshape(conv3, [-1, w.get_shape().as_list()[0]])
    fc = tf.add(tf.matmul(fc, w), bias)
    fc = tf.nn.relu(fc)
    fc = tf.nn.dropout(fc, keep_prob=0.5)

with tf.name_scope('output-layer'):
    w_out = tf.get_variable('W_OUT', shape=(128, n_classes), initializer=tf.contrib.layers.xavier_initializer())
    bias_out = tf.get_variable('B_OUT', shape=(n_classes), initializer=tf.contrib.layers.xavier_initializer())
    out = tf.add(tf.matmul(fc, w_out), bias_out)
    tf.add_to_collection('output', out)

with tf.name_scope('cost'):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out, labels=y))
    tf.add_to_collection('cost', cost)

with tf.name_scope('optimization'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    # optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9, use_nesterov=True).minimize(cost)

with tf.name_scope('predictions'):
    correct_prediction = tf.equal(tf.argmax(out, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.add_to_collection('accuracy', accuracy)

with tf.name_scope('summaries'):
    tf.summary.scalar('cost', cost)
    tf.summary.histogram('weights-fc', w)
    tf.summary.histogram('biases-fc', bias)
    tf.summary.histogram('activations-fc', fc)
    merged = tf.summary.merge_all()

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(init)
    train_loss = []
    test_loss = []
    train_accuracy = []
    test_accuracy = []
    summary_writer = tf.summary.FileWriter('/tmp/cnn-mnist-fashion', graph=tf.get_default_graph())
    for i in range(training_iters):
        for batch in range(len(train_X) // batch_size):
            batch_x = train_X[batch * batch_size:min((batch + 1) * batch_size, len(train_X))]
            batch_y = train_y[batch * batch_size:min((batch + 1) * batch_size, len(train_y))]
            # Run optimization op (backprop).
            # Calculate batch loss and accuracy
            _, summary, loss, acc = sess.run([optimizer, merged, cost, accuracy], feed_dict={x: batch_x, y: batch_y})
            summary_writer.add_summary(summary, i)
            # loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x, y: batch_y})
        print("Iter " + str(i + 1) + ", Loss= " + \
              "{:.6f}".format(loss) + ", Training Accuracy= " + \
              "{:.5f}".format(acc))

        # Calculate accuracy for all 10000 mnist test images
        test_acc, valid_loss = sess.run([accuracy, cost], feed_dict={x: test_X, y: test_y})
        train_loss.append(loss)
        test_loss.append(valid_loss)
        train_accuracy.append(acc)
        test_accuracy.append(test_acc)
        print("Testing Accuracy:", "{:.5f}".format(test_acc))
    summary_writer.close()
    saver.save(sess, os.path.join(data_dir, 'mnist_fashion.ckpt'))

plot_data(x=train_loss, y1=test_loss, title='Training and Test loss', xlabel='Epochs ', ylabel='Loss',
          xplabel='Training Loss', yplabel='Test Loss')
plot_data(x=train_accuracy, y1=train_accuracy, y2=test_accuracy, title='Training and Test Accuracy', xlabel='Epochs ',
          ylabel='Loss', xplabel='Training Accuracy', yplabel='Test Accuracy')

plt.close()
