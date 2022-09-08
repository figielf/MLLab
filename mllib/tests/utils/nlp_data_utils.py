import operator
import os
import string
from glob import glob

import numpy as np
from nltk.corpus import brown

from tests.utils.data_utils import get_data_dir

UNKNOWN_STR = '<UNKNOWN>'
START_STR = '<START>'
END_STR = '<END>'


def tokenize_brown_sentence(sentence):
    for token in sentence:
        yield token.lower()


def remove_punctuation(s):
    return s.translate(str.maketrans('', '', string.punctuation))


def generate_random_sequence(seq_len, V, start_seq_state, end_seq_state):
    p = np.ones(V)
    p[start_seq_state], p[end_seq_state] = 0, 0
    p = p / p.sum()
    return np.random.choice(V, size=seq_len, p=p)


def split_sentences(sentences, separator=' '):
    return [s.split(separator) for s in sentences]


def get_idx2word_mapping(word2idx):
    return {idx: word for word, idx in word2idx.items()}


def idx_sqe2sentence(idx_seq, idx2word, separator=' '):
    return separator.join([idx2word[idx] for idx in idx_seq])


def get_sequences_with_word2idx_from_brown_corpus(n_vocab=2000, include_start_end_in_vocab=True, keep_words=
{'king', 'man', 'queen', 'woman', 'italy', 'rome', 'france', 'paris', 'london', 'britain', 'england'}):
    sentences = brown.sents()

    if include_start_end_in_vocab:
        word2idx = {START_STR: 0, END_STR: 1}
        idx2word = [START_STR, END_STR]
        i = 2
        word_idx_count = {
            0: float('inf'),
            1: float('inf'),
        }
    else:
        word2idx = {}
        idx2word = []
        i = 0
        word_idx_count = {}

    indexed_sentences = []
    for sentence in sentences:
        indexed_sentence = []
        for token in tokenize_brown_sentence(sentence):
            if token not in word2idx:
                idx2word.append(token)
                word2idx[token] = i
                i += 1

            idx = word2idx[token]
            indexed_sentence.append(idx)

            # keep track of counts for later sorting
            word_idx_count[idx] = word_idx_count.get(idx, 0) + 1
        indexed_sentences.append(indexed_sentence)

    if n_vocab is None:
        return indexed_sentences, word2idx

    # restrict vocab size
    if keep_words is not None:
        for word in keep_words:
            word_idx_count[word2idx[word]] = float('inf')

    sorted_word_idx_count = sorted(word_idx_count.items(), key=operator.itemgetter(1), reverse=True)
    word2idx_small = {}
    new_idx = 0
    idx_new_idx_map = {}
    for idx, count in sorted_word_idx_count[:n_vocab]:
        word = idx2word[idx]
        # print(word, count)
        word2idx_small[word] = new_idx
        idx_new_idx_map[idx] = new_idx
        new_idx += 1
    # let 'unknown' be the last token
    word2idx_small[UNKNOWN_STR] = new_idx
    unknown = new_idx

    if include_start_end_in_vocab:
        assert (START_STR in word2idx_small)
        assert (END_STR in word2idx_small)
    if keep_words is not None:
        for word in keep_words:
            assert (word in word2idx_small)

    # map old idx to new idx
    sentences_small = []
    for sentence in indexed_sentences:
        if len(sentence) > 1:
            new_sentence = [idx_new_idx_map[idx] if idx in idx_new_idx_map else unknown for idx in sentence]
            sentences_small.append(new_sentence)

    return sentences_small, word2idx_small, sentences


def get_sequences_from_sentences_given_word2idx(sentences, word2idx, tokenize_method):
    indexed_sentences = []
    for sentence in sentences:
        sequence = []
        for token in tokenize_method(sentence):
            if token in word2idx:
                sequence.append(word2idx[token])
            else:
                sequence.append(word2idx[UNKNOWN_STR])
        indexed_sentences.append(sequence)
    return indexed_sentences


def get_sequences_with_word2idx_from_wiki_corpus(n_vocab=20000, return_sentences=False):
    files = glob(get_data_dir('large_files') + '\enwiki-preprocessed\enwiki*.txt')
    all_word_counts = {}
    for f in files:
        for line in open(f, encoding='utf-8'):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    for word in s:
                        if word not in all_word_counts:
                            all_word_counts[word] = 0
                        all_word_counts[word] += 1

    V = min(n_vocab, len(all_word_counts))
    all_word_counts = sorted(all_word_counts.items(), key=lambda x: x[1], reverse=True)

    top_words = [w for w, count in all_word_counts[:V - 1]] + [UNKNOWN_STR]
    word2idx = {w: i for i, w in enumerate(top_words)}
    unk = word2idx[UNKNOWN_STR]

    sentences = []
    sequences = []
    for f in files:
        for line in open(f, encoding='utf-8'):
            if line and line[0] not in '[*-|=\{\}':
                s = remove_punctuation(line).lower().split()
                if len(s) > 1:
                    if return_sentences:
                        sentences.append(s)
                    # if a word is not nearby another word, there won't be any context!
                    # and hence nothing to train!
                    seq = [word2idx[w] if w in word2idx else unk for w in s]
                    sequences.append(seq)
    return sequences, word2idx, sentences


def get_conll2000_data(split_sequences=False, index_start=0):
    train_data_path = os.path.join(get_data_dir('conll2000'), 'train.txt')
    test_data_path = os.path.join(get_data_dir('conll2000'), 'test.txt')

    word2idx = {}
    tag2idx = {}
    word_idx = index_start
    tag_idx = index_start
    Xtrain = []
    Ytrain = []
    currentX = []
    currentY = []
    for line in open(train_data_path):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag, _ = r
            if word not in word2idx:
                word2idx[word] = word_idx
                word_idx += 1
            currentX.append(word2idx[word])

            if tag not in tag2idx:
                tag2idx[tag] = tag_idx
                tag_idx += 1
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtrain.append(currentX)
            Ytrain.append(currentY)
            currentX = []
            currentY = []

    if not split_sequences:
        Xtrain = currentX
        Ytrain = currentY

    # load and score test data
    Xtest = []
    Ytest = []
    currentX = []
    currentY = []
    for line in open(test_data_path):
        line = line.rstrip()
        if line:
            r = line.split()
            word, tag, _ = r
            if word in word2idx:
                currentX.append(word2idx[word])
            else:
                currentX.append(word_idx)  # use this as unknown
            currentY.append(tag2idx[tag])
        elif split_sequences:
            Xtest.append(currentX)
            Ytest.append(currentY)
            currentX = []
            currentY = []
    if not split_sequences:
        Xtest = currentX
        Ytest = currentY

    return Xtrain, Ytrain, Xtest, Ytest, word2idx


class Tree:
    def __init__(self, word, label):
        self.left = None
        self.right = None
        self.word = word
        self.label = label


def display_tree(t, lvl=0):
    prefix = ''.join(['>']*lvl)
    if t.word is not None:
        print("%s%s %s" % (prefix, t.label, t.word))
    else:
        print("%s%s -" % (prefix, t.label))
        # if t.left is None or t.right is None:
        #     raise Exception("Tree node has no word but left and right child are None")
    if t.left:
        display_tree(t.left, lvl + 1)
    if t.right:
        display_tree(t.right, lvl + 1)


def get_trees_data_with_word2idx_from_ptb():
    # like the wikipedia dataset, I want to return 2 things:
    # word2idx mapping, sentences
    # here the sentences should be Tree objects

    def str2tree(s, word2idx, current_idx):
        # take a string that starts with ( and MAYBE ends with )
        # return the tree that it represents
        # EXAMPLE: "(3 (2 It) (4 (4 (2 's) (4 (3 (2 a) (4 (3 lovely) (2 film))) (3 (2 with) (4 (3 (3 lovely) (2 performances)) (2 (2 by) (2 (2 (2 Buy) (2 and)) (2 Accorsi))))))) (2 .)))"
        # NOTE: not every node has 2 children (possibly not correct ??)
        # NOTE: not every node has a word
        # NOTE: every node has a label
        # NOTE: labels are 0,1,2,3,4
        # NOTE: only leaf nodes have words
        # s[0] = (, s[1] = label, s[2] = space, s[3] = character or (

        # print "Input string:", s, "len:", len(s)

        # global current_idx
        label = int(s[1])
        if s[3] == '(':
            t = Tree(None, label)
            # try:

            # find the string that represents left child
            # it can include trailing characters we don't need, because we'll only look up to )
            child_s = s[3:]
            t.left, current_idx = str2tree(child_s, word2idx, current_idx)

            # find the string that represents right child
            # can contain multiple ((( )))
            # left child is completely represented when we've closed as many as we've opened
            # we stop at 1 because the first opening paren represents the current node, not children nodes
            i = 0
            depth = 0
            for c in s:
                i += 1
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 1:
                        break
            # print "index of right child", i

            t.right, current_idx = str2tree(s[i + 1:], word2idx, current_idx)

            # except Exception as e:
            #     print "Exception:", e
            #     print "Input string:", s
            #     raise e

            # if t.left is None or t.right is None:
            #     raise Exception("Tree node has no word but left and right child are None")
            return t, current_idx
        else:
            # this has a word, so it's a leaf
            r = s.split(')', 1)[0]
            word = r[3:].lower()
            # print "word found:", word

            if word not in word2idx:
                word2idx[word] = current_idx
                current_idx += 1

            t = Tree(word2idx[word], label)
            return t, current_idx

    data_folder = os.path.join(get_data_dir('sentiment_trees'), 'trees')
    train_data_path = os.path.join(data_folder, 'train.txt')
    test_data_path = os.path.join(data_folder, 'test.txt')
    dev_data_path = os.path.join(data_folder, 'dev.txt')

    word2idx = {}
    train = []
    test = []

    current_idx = 0
    # train set first
    for line in open(train_data_path):
        line = line.rstrip()
        if line:
            #print(f'Processing line:\n{line}')
            t, current_idx = str2tree(line, word2idx, current_idx)
            # if t.word is None and t.left is None and t.right is None:
            #     print "sentence:", line
            #display_tree(t)
            # print ""
            train.append(t)
            # break

    # test set
    for line in open(test_data_path):
        line = line.rstrip()
        if line:
            t, current_idx = str2tree(line, word2idx, current_idx)
            test.append(t)
    return train, test, word2idx
