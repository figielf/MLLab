import os
import re

import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences

from data_utils import get_data_dir, split_by_train_size


def get_toxic_data(train_size=0.8):
    # https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge
    # https://lazyprogrammer.me/course_files/toxic_comment_train.csv
    print("Reading in and transforming data...")
    df = pd.read_csv(get_data_dir(os.path.join('large_files', 'toxic_comment_train.csv')), encoding='utf-8')
    sentences = df["comment_text"].fillna("DUMMY_VALUE").values
    possible_labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    targets = df[possible_labels].values
    X_train, X_test, Y_train, Y_test = split_by_train_size(sentences, targets, train_size=train_size)
    return X_train, X_test, Y_train, Y_test, possible_labels


def get_spanish_english_translations_data(n_samples=10000):
    # http://www.manythings.org/anki/   -> spa_eng.zip

    input_texts = []  # sentence in original language
    target_texts = []  # sentence in target language
    target_texts_inputs = []  # sentence in target language offset by 1
    print("Reading in and transforming data...")
    with open(get_data_dir(os.path.join('large_files', 'spa.txt')), encoding='utf-8') as f:
        t = 0
        for line in f:
            # only keep a limited number of samples
            t += 1
            if t > n_samples:
                break

            # input and target are separated by tab
            if '\t' not in line:
                continue

            # split up the input and translation
            input_text, translation, *rest = line.rstrip().split('\t')

            # make the target input and output
            # recall we'll be using teacher forcing
            target_text = translation + ' <eos>'
            target_text_input = '<sos> ' + translation

            input_texts.append(input_text)
            target_texts.append(target_text)
            target_texts_inputs.append(target_text_input)

    return input_texts, target_texts, target_texts_inputs


def get_babi_qa_data(challenge_type, n_samples=10000):
    # https://research.fb.com/downloads/babi/   -> babi-tasks-v1-2.tar.gz
    # https://s3.amazonaws.com/text-datasets/babi_tasks_1-20_v1-2.tar.gz

    def tokenize(sent):
        '''Return the tokens of a sentence including punctuation.

        >>> tokenize('Bob dropped the apple. Where is the apple?')
        ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
        '''
        return [x.strip() for x in re.split('(\W+?)', sent) if x.strip()]

    def get_babi_stories(f):

        # data will return a list of triples
        # each triple contains:
        #   1. a story
        #   2. a question about the story
        #   3. the answer to the question
        data = []

        # use this list to keep track of the story so far
        story = []

        for line in f:
            line = line.strip()

            # split the line number from the rest of the line
            nid, line = line.split(' ', 1)

            # see if we should begin a new story
            if int(nid) == 1:
                story = []

            # this line contains a question and answer if it has a tab
            #       question<TAB>answer
            # it also tells us which line in the story is relevant to the answer
            # Note: we actually ignore this fact, since the model will learn
            #       which lines are important
            # Note: the max line number is not the number of lines of the story
            #       since lines with questions do not contain any story
            # one story may contain MULTIPLE questions
            if '\t' in line:
                q, a, supporting = line.split('\t')
                q = tokenize(q)

                # numbering each line is very useful
                # it's the equivalent of adding a unique token to the front
                # of each sentence
                story_so_far = [[str(i)] + s for i, s in enumerate(story) if s]

                # uncomment if you want to see what a story looks like
                # if not printed and np.random.rand() < 0.5:
                #     print("story_so_far:", story_so_far)
                #     printed = True
                data.append((story_so_far, q, a))
                story.append('')
            else:
                # just add the line to the current story
                story.append(tokenize(line))
        return data

    # convert stories from words into lists of word indexes (integers)
    # pad each sequence so that they are the same length
    # we will need to re-pad the stories later so that each story
    # is the same length
    def vectorize_stories(data, word2idx, story_maxlen, query_maxlen):
        inputs, queries, answers = [], [], []
        for story, query, answer in data:
            inputs.append([[word2idx[w] for w in s] for s in story])
            queries.append([word2idx[w] for w in query])
            answers.append([word2idx[answer]])
        return (
            [pad_sequences(x, maxlen=story_maxlen) for x in inputs],
            pad_sequences(queries, maxlen=query_maxlen),
            np.array(answers)
        )

    # recursively flatten a list
    def should_flatten(el):
        return not isinstance(el, (str, bytes))

    def flatten(l):
        for el in l:
            if should_flatten(el):
                yield from flatten(el)
            else:
                yield el

    # this is like 'pad_sequences' but for entire stories
    # we are padding each story with zeros so every story
    # has the same number of sentences
    # append an array of zeros of size:
    # (max_sentences - num sentences in story, max words in sentence)
    def stack_inputs(inputs, story_maxsents, story_maxlen):
        for i, story in enumerate(inputs):
            inputs[i] = np.concatenate(
                [
                    story,
                    np.zeros((story_maxsents - story.shape[0], story_maxlen), 'int')
                ]
            )
        return np.stack(inputs)

    challenges = {
        # QA1 with 10,000 samples
        'single_supporting_fact_10k': 'babi_tasks_1-20_v1-2\en-10k\qa1_single-supporting-fact_{}.txt',
        # QA2 with 10,000 samples
        'two_supporting_facts_10k': 'babi_tasks_1-20_v1-2\en-10k\qa2_two-supporting-facts_{}.txt',
    }

    # input should either be 'single_supporting_fact_10k' or 'two_supporting_facts_10k'
    challenge = challenges[challenge_type]

    print("Reading in and transforming data...")
        # returns a list of triples of:
        # (story, question, answer)
        # story is a list of sentences
        # question is a sentence
        # answer is a word
    with open(get_data_dir(os.path.join('large_files', challenge.format('train'))), encoding='utf-8') as f:
        train_stories = get_babi_stories(f)
    with open(get_data_dir(os.path.join('large_files', challenge.format('test'))), encoding='utf-8') as f:
        test_stories = get_babi_stories(f)

    # group all the stories together
    stories = train_stories + test_stories

    # so we can get the max length of each story, of each sentence, and of each question
    story_maxlen = max((len(s) for x, _, _ in stories for s in x))
    story_maxsents = max((len(x) for x, _, _ in stories))
    query_maxlen = max(len(x) for _, x, _ in stories)

    # Create vocabulary of corpus and find size, including a padding element.
    vocab = sorted(set(flatten(stories)))
    vocab.insert(0, '<PAD>')
    vocab_size = len(vocab)

    # Create an index mapping for the vocabulary.
    word2idx = {c: i for i, c in enumerate(vocab)}

    # convert stories from strings to lists of integers
    inputs_train, queries_train, answers_train = vectorize_stories(
        train_stories,
        word2idx,
        story_maxlen,
        query_maxlen
    )
    inputs_test, queries_test, answers_test = vectorize_stories(
        test_stories,
        word2idx,
        story_maxlen,
        query_maxlen
    )

    # convert inputs into 3-D numpy arrays
    inputs_train = stack_inputs(inputs_train, story_maxsents, story_maxlen)
    inputs_test = stack_inputs(inputs_test, story_maxsents, story_maxlen)
    print("inputs_train.shape, inputs_test.shape", inputs_train.shape, inputs_test.shape)

    return train_stories, test_stories, \
           inputs_train, queries_train, answers_train, \
           inputs_test, queries_test, answers_test, \
           story_maxsents, story_maxlen, query_maxlen, \
           vocab, vocab_size, word2idx


def get_airline_tweets_data(train_size=0.8):
    # https://www.kaggle.com/crowdflower/twitter-airline-sentiment
    # https://lazyprogrammer.me/course_files/AirlineTweets.csv
    print("Reading in and transforming data...")
    df = pd.read_csv(get_data_dir('AirlineTweets.csv'))
    df = df[df['airline_sentiment'] != 'neutral']
    tweets = df['text'].values
    sentiments = df['airline_sentiment'].values
    X_train, X_test, Y_train, Y_test = split_by_train_size(tweets, sentiments, train_size=train_size)
    return X_train, X_test, Y_train, Y_test, df


def get_robert_frost_data():
    lines = []
    with open(get_data_dir('robert_frost.txt'), encoding='utf-8') as file:
        for line in file:
            txt = line.rstrip()
            if len(line) > 0:
                lines.append(line)
    return lines


def get_bbc_text_data(train_size=0.8):
    df = pd.read_csv(get_data_dir('bbc_text_cls.csv'))
    texts = df['text'].values
    targets = df['labels'].values
    X_train, X_test, Y_train, Y_test = split_by_train_size(texts, targets, train_size=train_size)
    return X_train, X_test, Y_train, Y_test