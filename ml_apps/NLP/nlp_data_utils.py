import os

import pandas as pd

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
