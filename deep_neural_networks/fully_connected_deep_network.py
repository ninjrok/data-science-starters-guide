import pickle
import os

import tensorflow as tf
from sklearn.metrics import accuracy_score

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
n_hidden = 50
learning_rate = 0.001
n_epochs = 10
batch_size = 100

print(train_y.shape)

with tf.name_scope('placeholders'):
    X = tf.placeholder(tf.float32, (None, n_features))
    y = tf.placeholder(tf.float32, (None,))

with tf.name_scope('hidden-layer'):
    W = tf.Variable(tf.random_normal((n_features, n_hidden)))
    b = tf.Variable(tf.random_normal((n_hidden,)))
    x_hidden = tf.nn.relu(tf.matmul(X, W) + b)

with tf.name_scope('output'):
    W = tf.Variable(tf.random_normal((n_hidden, 1)))
    b = tf.Variable(tf.random_normal((1,)))
    y_logit = tf.matmul(x_hidden, W) + b
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

train_writer = tf.summary.FileWriter('/tmp/fcnet-tox21', tf.get_default_graph())

N = train_X.shape[0]
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    step = 0
    for epoch in range(n_epochs):
        pos = 0
        while pos < N:
            batch_X = train_X[pos:pos + batch_size]
            batch_y = train_y[pos:pos + batch_size]
            feed_dict = {X: batch_X, y: batch_y}
            _, summary, loss_val = sess.run([optimizer, merged, loss], feed_dict=feed_dict)
            print("epoch %d, step %d, loss: %f" % (epoch, step, loss_val))
            train_writer.add_summary(summary, step)

            step += 1
            pos += batch_size

    valid_y_pred = sess.run(y_pred, feed_dict={X: valid_X})

score = accuracy_score(valid_y, valid_y_pred)
print("Unweighted Classification Accuracy: %f" % score)

weighted_score = accuracy_score(valid_y, valid_y_pred, sample_weight=valid_w)
print("Weighted Classification Accuracy: %f" % weighted_score)
