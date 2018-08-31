import pickle
import os

import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix

data_directory = os.path.join(os.environ['VIRTUAL_ENV'], '..', 'data', 'tox21')

data = {}
for pickle_file in os.listdir(data_directory):
    with open(os.path.join(data_directory, pickle_file), 'rb') as reader:
        data[pickle_file.strip('.pkl')] = pickle.load(reader)

train_X, train_y, train_w = data['train_X'], data['train_y'], data['train_w']
valid_X, valid_y, valid_w = data['valid_X'], data['valid_y'], data['valid_w']
test_X, test_y, test_w = data['test_X'], data['test_y'], data['test_w']

# Remove extra tasks
train_y = train_y[:, 0]
valid_y = valid_y[:, 0]
test_y = test_y[:, 0]
train_w = train_w[:, 0]
valid_w = valid_w[:, 0]
test_w = test_w[:, 0]

n_features = 1024
n_hidden = 150
n_hidden2 = 75
learning_rate = 0.001
n_epochs = 10
batch_size = 100
dropout_prob = 1.0

print(train_y.shape)

with tf.name_scope('placeholders'):
    X = tf.placeholder(tf.float32, (None, n_features))
    y = tf.placeholder(tf.float32, (None,))
    keep_prob = tf.placeholder(tf.float32)

with tf.name_scope('hidden-layer-1'):
    W = tf.Variable(tf.random_normal((n_features, n_hidden)))
    b = tf.Variable(tf.random_normal((n_hidden,)))
    x_hidden = tf.nn.relu(tf.matmul(X, W) + b)
    x_hidden = tf.nn.dropout(x_hidden, keep_prob=keep_prob)

with tf.name_scope('hidden-layer-2'):
    W = tf.Variable(tf.random_normal((n_hidden, n_hidden2)))
    b = tf.Variable(tf.random_normal((n_hidden2,)))
    x_hidden2 = tf.nn.relu(tf.matmul(x_hidden, W) + b)
    x_hidden2 = tf.nn.dropout(x_hidden2, keep_prob=keep_prob)

with tf.name_scope('output'):
    W = tf.Variable(tf.random_normal((n_hidden2, 1)))
    b = tf.Variable(tf.random_normal((1,)))
    y_logit = tf.matmul(x_hidden2, W) + b
    y_prob = tf.sigmoid(y_logit)
    y_pred = tf.round(y_prob)

with tf.name_scope('loss'):
    y_expand = tf.expand_dims(y, 1)
    entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=y_logit, labels=y_expand)
    loss = tf.reduce_sum(entropy)

with tf.name_scope('optimization'):
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

with tf.name_scope('summaries'):
    tf.summary.scalar('loss', loss)
    merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter('/tmp/fcnet-tox21-dropout', tf.get_default_graph())

N = train_X.shape[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    for epoch in range(n_epochs):
        pos = 0
        while pos < N:
            batch_X = train_X[pos:pos + batch_size]
            batch_y = train_y[pos:pos + batch_size]
            feed_dict = {X: batch_X, y: batch_y, keep_prob: dropout_prob}
            _, summary, loss_val = sess.run([optimizer, merged, loss], feed_dict=feed_dict)
            print("epoch %d, step %d, loss: %f" % (epoch, step, loss_val))
            train_writer.add_summary(summary, step)

            step += 1
            pos += batch_size

    valid_y_pred = sess.run(y_pred, feed_dict={X: valid_X, keep_prob: 1.0})
    test_y_pred = sess.run(y_pred, feed_dict={X: test_X, keep_prob: 1.0})

score = accuracy_score(test_y, test_y_pred, sample_weight=test_w)
print('Weighted classification accuracy(test set): {}'.format(score))

weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
print("Weighted Classification Accuracy(validation set): %f" % weighted_score)

conf_matrix = confusion_matrix(test_y, test_y_pred)
print('Confusion Matrix of test set: \n{}'.format(conf_matrix))
