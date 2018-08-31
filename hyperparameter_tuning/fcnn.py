import os
import pickle
import uuid

import tensorflow as tf

from sklearn.metrics import accuracy_score


class FCNN:
    def __init__(self):
        self.data_directory = os.path.join(os.environ['VIRTUAL_ENV'], '..', 'data', 'tox21')

        data = {}
        for pickle_file in os.listdir(self.data_directory):
            with open(os.path.join(self.data_directory, pickle_file), 'rb') as reader:
                data[pickle_file.strip('.pkl')] = pickle.load(reader)

        self.train_X, self.train_y, self.train_w = data['train_X'], data['train_y'], data['train_w']
        self.valid_X, self.valid_y, self.valid_w = data['valid_X'], data['valid_y'], data['valid_w']
        self.test_X, self.test_y, self.test_w = data['test_X'], data['test_y'], data['test_w']

        # Remove extra tasks
        self.train_y = self.train_y[:, 0]
        self.valid_y = self.valid_y[:, 0]
        self.test_y = self.test_y[:, 0]
        self.train_w = self.train_w[:, 0]
        self.valid_w = self.valid_w[:, 0]
        self.test_w = self.test_w[:, 0]

    def train(self, n_layers=1, n_epochs=10, dropout_prob=1.0, batch_size=100, learning_rate=0.001, n_hidden=150):
        n_features = self.train_X.shape[1]
        uid = uuid.uuid4()

        with tf.Graph().as_default() as graph:
            with tf.name_scope('placeholders'):
                x = tf.placeholder(tf.float32, (None, n_features))
                y = tf.placeholder(tf.float32, (None,))
                keep_prob = tf.placeholder(tf.float32)

            for layer in range(n_layers):
                with tf.name_scope('hidden-layer-{}'.format(layer)):
                    if layer == 0:
                        w = tf.Variable(tf.random_normal((n_features, n_hidden)))
                        b = tf.Variable(tf.random_normal((n_hidden,)))
                        x_hidden = tf.nn.relu(tf.matmul(x, w) + b)
                    else:
                        w = tf.Variable(tf.random_normal((n_hidden, n_hidden)))
                        b = tf.Variable(tf.random_normal((n_hidden,)))
                        x_hidden = tf.nn.relu(tf.matmul(x_hidden, w) + b)
                    x_hidden = tf.nn.dropout(x_hidden, keep_prob=keep_prob)

            with tf.name_scope('output'):
                w = tf.Variable(tf.random_normal((n_hidden, 1)))
                b = tf.Variable(tf.random_normal((1,)))
                y_logit = tf.matmul(x_hidden, w) + b
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

            train_writer = tf.summary.FileWriter('/tmp/fcnet-{}-{}'.format(uid, n_layers), graph=graph)

            n = self.train_X.shape[0]
            with tf.Session(graph=graph) as sess:
                sess.run(tf.global_variables_initializer())
                step = 0
                for epoch in range(n_epochs):
                    pos = 0
                    while pos < n:
                        batch_X = self.train_X[pos:pos + batch_size]
                        batch_y = self.train_y[pos:pos + batch_size]
                        feed_dict = {x: batch_X, y: batch_y, keep_prob: dropout_prob}
                        _, summary, loss_val = sess.run([optimizer, merged, loss], feed_dict=feed_dict)
                        print("epoch %d, step %d, loss: %f" % (epoch, step, loss_val))
                        train_writer.add_summary(summary, step)

                        step += 1
                        pos += batch_size

                train_y_pred = sess.run(y_pred, feed_dict={x: self.train_X, keep_prob: 1.0})
                valid_y_pred = sess.run(y_pred, feed_dict={x: self.valid_X, keep_prob: 1.0})
                test_y_pred = sess.run(y_pred, feed_dict={x: self.test_X, keep_prob: 1.0})

        train_set_accuracy = accuracy_score(self.train_y, train_y_pred, sample_weight=self.train_w)
        test_set_accuracy = accuracy_score(self.test_y, test_y_pred, sample_weight=self.test_w)
        validation_set_accuracy = accuracy_score(self.valid_y, valid_y_pred, sample_weight=self.valid_w)

        return uid, train_set_accuracy, test_set_accuracy, validation_set_accuracy


if __name__ == '__main__':
    fcnn = FCNN()
    uid, train_acc, test_acc, val_acc = fcnn.train(n_epochs=1050, dropout_prob=1.0, n_hidden=50, n_layers=20)
    print('Accuracy:\nTrain: {}\nValidation: {}\nTest: {}'.format(train_acc, val_acc, test_acc))
