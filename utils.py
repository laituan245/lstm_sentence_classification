import numpy as np

import re
import json
import unicodedata

from io import open

def normalize_string(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def one_hot_encoding(num_classes, idx):
    encoding = [0] * num_classes
    encoding[idx] = 1
    return encoding

def load_glove_vectors(path, dictionary, emb_size = 300):
    vocab_size = dictionary.n_words
    word2index = dictionary.word2index
    weights = np.random.rand(vocab_size, emb_size)

    with open(path, encoding='utf-8') as f:
        for line in f:
            word, emb = line.strip().split(' ', 1)
            if word in word2index:
                weights[word2index[word]] = np.asarray(list(map(float, emb.split(' ')))[:emb_size])
    return weights

def indexes_from_sentence(sentence, dictionary, max_len = None):
    word2index = dictionary.word2index
    SOS_token = dictionary.SOS_token
    EOS_token = dictionary.EOS_token
    UNK_token = dictionary.UNK_token
    indexes = []
    for word in word_tokenize(normalize_string(sentence)):
        if len(word) > 0:
            if word in word2index:
                indexes.append(word2index[word])
            else:
                indexes.append(UNK_token)
    indexes = [SOS_token] + indexes + [EOS_token]
    if max_len:
        while len(indexes) < max_len:
            indexes.append(EOS_token)
    return indexes

def unicode_to_ascii(s):
    s = unicode(s) # For Python 2.7
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def word_tokenize(sent):
    return sent.split(' ')

# Helper classes
class Dictionary:
    SOS_token = 0
    EOS_token = 1
    UNK_token = 2

    def __init__(self):
        # Intialization
        self.word2index = {}
        self.index2word = {0: "[SOS]", 1: "[EOS]", 2: "[UNK]"}
        self.n_words = 3

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def add_words(self, words):
        for word in words:
            self.add_word(word)

class AugmentedList:
    def __init__(self, items):
        self.items = items
        self.cur_idx = 0

    def next_items(self, batch_size):
        items = self.items
        start_idx = self.cur_idx
        end_idx = start_idx + batch_size
        if end_idx <= self.size:
            self.cur_idx = end_idx % self.size
            return items[start_idx : end_idx]
        else:
            first_part = items[start_idx : self.size]
            remain_size = batch_size - (self.size - start_idx)
            second_part = items[0 : remain_size]
            self.cur_idx = remain_size
            return first_part + second_part

    @property
    def size(self):
        return len(self.items)
