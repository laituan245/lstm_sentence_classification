from __future__ import print_function

import tensorflow as tf
import random

from utils import AugmentedList, Dictionary
from utils import normalize_string, word_tokenize
from utils import one_hot_encoding, indexes_from_sentence

from io import open

TRAIN_MODE = tf.contrib.learn.ModeKeys.TRAIN
EVAL_MODE = tf.contrib.learn.ModeKeys.EVAL
POSITIVE_REVIEWS_FILE = './data/rt-polarity.pos'
NEGAGIVE_REVIEWS_FILE = './data/rt-polarity.neg'
random.seed(8888)

class ReviewDataset:
    def __init__(self):
        self.data = []
        self.dictionary = Dictionary()
        self.max_sent_len = 0

        # Read the positive reviews
        with open(POSITIVE_REVIEWS_FILE, encoding='utf-8') as f:
            positive_reviews = f.readlines()
        for review in positive_reviews:
            review = normalize_string(review)
            review_words = word_tokenize(review)
            self.dictionary.add_words(review_words)
            self.data.append((review, 1))
            self.max_sent_len = max(self.max_sent_len, 2 + len(review_words))

        # Read the negative reviews
        with open(NEGAGIVE_REVIEWS_FILE, encoding='utf-8') as f:
            negative_reviews = f.readlines()
        for review in negative_reviews:
            review = normalize_string(review)
            review_words = word_tokenize(review)
            self.dictionary.add_words(review_words)
            self.data.append((review, 0))
            self.max_sent_len = max(self.max_sent_len, 2 + len(review_words))

        # Split the original dataset into train/test
        random.shuffle(self.data)
        split_index = int(0.9 * len(self.data))
        self.train = AugmentedList(self.data[:split_index])
        self.test = AugmentedList(self.data[split_index:])

    def next_batch(self, batch_size, mode = TRAIN_MODE):
        reviews, targets = [], []
        data = self.train if mode == TRAIN_MODE else self.test
        batch = data.next_items(batch_size)
        for (review, target) in batch:
            review = indexes_from_sentence(review, self.dictionary, self.max_sent_len)
            target = one_hot_encoding(2, target)
            reviews.append(review)
            targets.append(target)
        return reviews, targets
