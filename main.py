import tensorflow as tf
import numpy as np
import sys

from model import Model
from reader import ReviewDataset, TRAIN_MODE, EVAL_MODE
from utils import load_glove_vectors

# Constants
GLOVE_VECTORS_FILE = './data/glove.840B.300d.txt'
EMBEDDING_DIM = 300
LEARNING_RATE = 0.01
BATCH_SIZE = 32
NUMBER_ITERATIONS = 20000
LOGS_PATH = './logs/'

# Initialization
dataset = ReviewDataset()
dictionary = dataset.dictionary
weights = load_glove_vectors(GLOVE_VECTORS_FILE, dictionary, emb_size=EMBEDDING_DIM)
model = Model(max_sent_len = dataset.max_sent_len, num_labels = 2,
              vocab_size = dictionary.n_words, embedding_dim = EMBEDDING_DIM,
              learning_rate = LEARNING_RATE)
saver = tf.train.Saver()

# Start TensorFlow
with tf.Session() as sess:
    sess.run(model.init)

    # create log writer object
    writer = tf.summary.FileWriter(LOGS_PATH, graph=tf.get_default_graph())

    sess.run(model.embedding_init, feed_dict={model.embedding_placeholder: weights})
    train_accuracies = []
    for ix in range(NUMBER_ITERATIONS):
        batch = dataset.next_batch(BATCH_SIZE, TRAIN_MODE)
        _, batch_accuracy = model.train(sess, batch)
        train_accuracies.append(batch_accuracy)
        # Write summary to logs
        summary = sess.run(model.summary_op)
        writer.add_summary(summary, ix)
        if ix > 0 and ix % 500 == 0:
            # Evaluate the model on the test set
            batch = dataset.next_batch(dataset.test.size, EVAL_MODE)
            nb_corrects = model.evaluate(sess, batch)
            test_accuracy = float(nb_corrects) / dataset.test.size
            # Print out information
            print('===> Iter {}: | Train acc {:.4f} | Test acc {:.4f}'
                  .format(ix, np.mean(train_accuracies[-500:]), test_accuracy))
            sys.stdout.flush()
            # Save the model to disk
            saver.save(sess, './models/my_first_model')
            print('Saved the lastest model to disk\n')
