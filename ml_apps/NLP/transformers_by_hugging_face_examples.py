import textwrap
from datetime import datetime
import pickle

import numpy as np
from transformers import pipeline

from NLP.nlp_data_utils import get_robert_frost_data, get_bbc_text_data
from NLP.nlp_data_utils import get_airline_tweets_data


def run_sentiment_analysis_task():
    classifier = pipeline('sentiment-analysis')
    print(classifier('There is something strange to work on'))

    X_train, _, Y_train, _, _ = get_airline_tweets_data(train_size=1.)

    t0 = datetime.now()
    predictions = classifier(X_train[:10].tolist())
    print(f'Analyzed sentiments for {len(predictions)} texts in {datetime.now() - t0} seconds')
    print(predictions)


def run_text_generation_task():
    generator = pipeline("text-generation")
    print(generator(['There is something strange to work on']))

    text_lines = get_robert_frost_data()

    t0 = datetime.now()
    predictions = generator(text_lines[:10])
    print(f'Generated texts for {len(predictions)} texts in {datetime.now() - t0} seconds')
    print(predictions)


def run_masked_language_modeling_task():
    demask = pipeline('fill-mask')
    print(demask(['There is something strange to <mask> on']))

    text = 'Shares in train and plane-making giant Bombardier have fallen to a 10-year low following the <mask> ' + \
           'of its chief executive and two members of the board.'

    t0 = datetime.now()
    predictions = demask([text])
    print(f'Generated texts for {len(predictions)} texts in {datetime.now() - t0} seconds')
    print(predictions)


def run_entity_named_recognition_task():
    ner = pipeline("ner", aggregation_strategy='simple', device=0)
    print(ner(['There is something strange to <mask> on']))

    text = 'He was well backed by England hopeful Mark Butcher who made 70 as Surrey closed on 429 for seven, a lead of 234.'

    t0 = datetime.now()
    predictions = ner([text])
    print(f'Check NER in text in {datetime.now() - t0} seconds')
    print(predictions)


if __name__ == '__main__':
    # do tasks using Hugging Face pipelines
    #run_sentiment_analysis_task()
    #run_text_generation_task()
    #run_masked_language_modeling_task()
    run_entity_named_recognition_task()
