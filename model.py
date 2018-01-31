from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

def conv_relu(input, kernel_shape, bias_shape):
    # Create variable named "weights".
    weights = tf.get_variable("weights", kernel_shape,
        initializer=tf.random_normal_initializer())
    # Create variable named "biases".
    biases = tf.get_variable("biases", bias_shape,
        initializer=tf.constant_initializer(0.1))
    conv = tf.nn.conv2d(input, weights,
        strides=[1, 1, 1, 1], padding='VALID')
    return tf.nn.relu(conv + biases)

class Model:
    def __init__(self, max_sent_len, num_labels, vocab_size, embedding_dim,
                 learning_rate):
        rnn_size = 150

        with tf.name_scope('placeholders'):
            self.keep_prob = tf.placeholder('float32', shape=(), name='drop')
            self.sentence_length = tf.placeholder(tf.int32, shape=(None))
            self.sentence = tf.placeholder(tf.int32, shape=(None, max_sent_len))
            self.target = tf.placeholder(tf.float32, shape=(None, num_labels))

        with tf.variable_scope('embedding'):
            embedding = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_dim]), trainable=True)
            self.embedding_placeholder = tf.placeholder(tf.float32, [vocab_size, embedding_dim])
            self.embedding_init = embedding.assign(self.embedding_placeholder)

            sent_emb = tf.nn.embedding_lookup(embedding, self.sentence)

        with tf.variable_scope('networks'):
            lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size)
            outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, sent_emb, sequence_length=self.sentence_length, dtype=tf.float32)
            features = final_state.h
            features_drop = tf.nn.dropout(features, self.keep_prob)

        with tf.variable_scope("predictions"):
            final_w = tf.Variable(tf.random_normal([rnn_size, num_labels]))
            final_b = tf.Variable(tf.random_normal([num_labels]))
            self.scores = tf.nn.xw_plus_b(features_drop, final_w, final_b)
            self.prediction = tf.argmax(self.scores, 1)

        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels = self.target, logits = self.scores)
            self.loss = tf.reduce_mean(losses)

        with tf.name_scope("accuracy"):
            correct_preds = tf.equal(tf.argmax(self.target, 1), self.prediction)
            self.accuracy = tf.reduce_mean(tf.cast(correct_preds, tf.float32))
            self.correct = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

        with tf.name_scope("optimization"):
            optimizer = tf.train.AdamOptimizer(learning_rate)
            gvs = optimizer.compute_gradients(self.loss)
            self.optimize = optimizer.apply_gradients(gvs)

        # Operation for initializing the variables
        self.init = tf.global_variables_initializer()

        # Operation to save and restore all the variables.
        self.saver = tf.train.Saver()

    def train(self, sess, batch, keep_prob = 0.8):
        batch_lengths, batch_sentences, batch_targets = batch
        _, batch_loss, batch_accuracy = sess.run(
                        [self.optimize, self.loss, self.accuracy],
                        feed_dict = {
                            self.keep_prob: keep_prob,
                            self.sentence_length: batch_lengths,
                            self.sentence: batch_sentences,
                            self.target: batch_targets,
                        })
        return batch_loss, batch_accuracy

    def evaluate(self, sess, batch):
        batch_lengths, batch_sentences, batch_targets = batch
        batch_correct = sess.run(
                        self.correct,
                        feed_dict = {
                            self.keep_prob: 1,
                            self.sentence_length: batch_lengths,
                            self.sentence: batch_sentences,
                            self.target: batch_targets,
                        })
        return batch_correct
