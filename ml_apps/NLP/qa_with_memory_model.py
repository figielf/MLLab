import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Dense, Embedding, Input, Lambda, Reshape, Dot, Add, Activation
from tensorflow.keras.optimizers import Adam, RMSprop

from NLP.glove_utils import load_glove_embeddings, get_embedding_matrix
from NLP.nlp_data_utils import get_spanish_english_translations_data, get_babi_qa_data


def print_weights(data, model, debug_model, scenario):
    (train_stories, test_stories, inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test,
     story_maxsents, story_maxlen, query_maxlen, vocab, vocab_size, word2idx) = data
    story_idx = np.random.choice(len(train_stories))

    # get weights from debug model
    i = inputs_train[story_idx:story_idx + 1]
    q = queries_train[story_idx:story_idx + 1]

    answer_pred = model.predict([i, q])
    story, question, ans = train_stories[story_idx]

    print('\nstory:')
    if scenario == 'single_supporting_fact_10k':
        w = debug_model.predict([i, q])
        w = w.flatten()
        for i, line in enumerate(story):
            print('{:1.5f}'.format(w[i]), '\t', ' '.join(line))
    elif scenario == 'two_supporting_facts_10k':
        w1, w2 = debug_model.predict([i, q])
        w1, w2 = w1.flatten(), w2.flatten()
        for i, line in enumerate(story):
            print('{:1.5f}'.format(w1[i]), '\t', '{:1.5f}'.format(w2[i]), '\t', ' '.join(line))
    else:
        raise Exception(f"Provided scenario '{scenario}' is not supported!")

    print('question:', ' '.join(question))
    print('answer:', ans)

    print('predicted answer:', vocab[np.argmax(answer_pred[0])])


def print_example_of_data(data):
    (train_stories, test_stories, inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test,
     story_maxsents, story_maxlen, query_maxlen, vocab, vocab_size, word2idx) = data
    story_idx = np.random.choice(len(train_stories))
    story, question, ans = train_stories[story_idx]
    print('\nstory:')
    for j, line in enumerate(story):
        print(' '.join(line))

    print('question:', ' '.join(question))
    print('answer:', ans)


def plot_history(history):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()

    # accuracies
    plt.subplot(1, 2, 2)
    plt.plot(history['accuracy'], label='acc')
    plt.plot(history['val_accuracy'], label='val_acc')
    plt.legend()
    plt.show()


def fit_model(model, data, desc):

    (train_stories, test_stories, inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test,
     story_maxsents, story_maxlen, query_maxlen, vocab, vocab_size, word2idx) = data

    print(f'Training {desc} model...')
    r = model.fit(
        [inputs_train, queries_train],
        answers_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=([inputs_test, queries_test], answers_test)
    )
    plot_history(r.history)


def bow_embedded(x, sum_axis, vocab_size, embedding_dim):
    embedding = Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM)(x)
    bow = tf.math.reduce_sum(embedding, axis=sum_axis)
    # bow = Lambda(lambda x: tf.math.reduce_sum(x, axis=2))(embedding)
    return bow

def build_single_supporting_fact_models(data):
    # Ns = story_maxsents  # number of sentences per story
    # Ts = story_maxlen  # story sentence length
    # Tq = query_maxlen  # question sentence length
    # Ta = 1  # answer sentence length
    # M = LATENT_DIM

    (train_stories, test_stories, inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test,
     story_maxsents, story_maxlen, query_maxlen, vocab, vocab_size, word2idx) = data

    i_story_sentence = Input(shape=(story_maxsents, story_maxlen), name='story_input')  # output shape=(BATCH_SIZE, Ns, Ts)
    i_question = Input(shape=(query_maxlen, ), name='question_input')  # output shape=(BATCH_SIZE, Tq)

    story_sentences_bow = bow_embedded(i_story_sentence, sum_axis=2, vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM)  # output shape=(BATCH_SIZE, Ns, EMBEDDING_DIM)
    question_bow = bow_embedded(i_question, sum_axis=1, vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM)  # output shape=(BATCH_SIZE, EMBEDDING_DIM)
    question_bow = Reshape((1, EMBEDDING_DIM), name='question_bow_reshape')(question_bow)  # output shape=(BATCH_SIZE, 1, EMBEDDING_DIM)

    qa_dotted = Dot(axes=2, name='qa_dotted')([story_sentences_bow, question_bow])  # output shape=(BATCH_SIZE, Ns, 1)
    qa_dotted = Reshape((story_maxsents, ), name='qa_dotted_reshape')(qa_dotted)  # output shape=(BATCH_SIZE, Ns)
    story_sentences_weights = Activation('softmax', name='story_sentences_weights_softmax')(qa_dotted)  # output shape=(BATCH_SIZE, Ns)
    story_sentences_weights = Reshape((story_maxsents, 1), name='story_sentences_weights_reshape')(story_sentences_weights)  # output shape=(BATCH_SIZE, Ns, 1)

    question_relevant_story = Dot(axes=1, name='story_summary_by_weighted_sentences')([story_sentences_weights, story_sentences_bow])  # output shape=(BATCH_SIZE, EMBEDDING_DIM, 1)
    question_relevant_story = Reshape((EMBEDDING_DIM, ), name='answer_reshape')(question_relevant_story)  # output shape=(BATCH_SIZE, EMBEDDING_DIM)

    answer = Dense(vocab_size, activation='softmax', name='dense_answer')(question_relevant_story)  # output shape=(BATCH_SIZE, 1, vocab_size)

    model = Model(inputs=[i_story_sentence, i_question], outputs=[answer])

    model.compile(
      optimizer=RMSprop(lr=1e-2),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
    )

    print('Single Supporting Fact model details:')
    print(model.summary())

    debug_model = Model([i_story_sentence, i_question], story_sentences_weights)

    return model, debug_model


def build_two_supporting_facts_models(data):
    # Ns = story_maxsents  # number of sentences per story
    # Ts = story_maxlen  # story sentence length
    # Tq = query_maxlen  # question sentence length
    # Ta = 1  # answer sentence length
    # M = LATENT_DIM

    (train_stories, test_stories, inputs_train, queries_train, answers_train, inputs_test, queries_test, answers_test,
     story_maxsents, story_maxlen, query_maxlen, vocab, vocab_size, word2idx) = data

    i_story_sentence = Input(shape=(story_maxsents, story_maxlen), name='story_input')  # output shape=(BATCH_SIZE, Ns, Ts)
    i_question = Input(shape=(query_maxlen, ), name='question_input')  # output shape=(BATCH_SIZE, Tq)

    story_sentences_bow = bow_embedded(i_story_sentence, sum_axis=2, vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM)  # output shape=(BATCH_SIZE, Ns, EMBEDDING_DIM)
    question_bow = bow_embedded(i_question, sum_axis=1, vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM)  # output shape=(BATCH_SIZE, EMBEDDING_DIM)

    hop_output_dense = Dense(EMBEDDING_DIM, activation='elu', name='hop_output_dense')

    def hop(query, story, name_suffix=''):
        query_reshaped = Reshape((1, EMBEDDING_DIM), name=f'question_bow_reshape{name_suffix}')(query)  # output shape=(BATCH_SIZE, 1, EMBEDDING_DIM)

        qa_dotted = Dot(axes=2, name=f'qa_dotted{name_suffix}')([story, query_reshaped])  # output shape=(BATCH_SIZE, Ns, 1)

        qa_dotted = Reshape((story_maxsents, ), name=f'qa_dotted_reshape{name_suffix}')(qa_dotted)  # output shape=(BATCH_SIZE, Ns)
        story_sentences_weights = Activation('softmax', name=f'story_sentences_weights_softmax{name_suffix}')(qa_dotted)  # output shape=(BATCH_SIZE, Ns)
        story_sentences_weights = Reshape((story_maxsents, 1), name=f'story_sentences_weights_reshape{name_suffix}')(story_sentences_weights)  # output shape=(BATCH_SIZE, Ns, 1)

        new_story_sentences_bow = bow_embedded(i_story_sentence, sum_axis=2, vocab_size=vocab_size, embedding_dim=EMBEDDING_DIM)  # output shape=(BATCH_SIZE, Ns, EMBEDDING_DIM)

        question_relevant_story = Dot(axes=1, name=f'story_summary_by_weighted_sentences{name_suffix}')([story_sentences_weights, new_story_sentences_bow])  # output shape=(BATCH_SIZE, EMBEDDING_DIM, 1)
        question_relevant_story = Reshape((EMBEDDING_DIM, ), name=f'answer_reshape{name_suffix}')(question_relevant_story)  # output shape=(BATCH_SIZE, EMBEDDING_DIM)

        hop_output = hop_output_dense(question_relevant_story)  # output shape=(BATCH_SIZE, EMBEDDING_DIM)
        return hop_output, new_story_sentences_bow, story_sentences_weights

    hop1_output, story_sentences_bow, hop1_weights = hop(question_bow, story_sentences_bow, '1')
    hop2_output, _, hop2_weights = hop(hop1_output, story_sentences_bow, '2')

    answer = Dense(vocab_size, activation='softmax', name='dense_answer')(hop2_output)  # output shape=(BATCH_SIZE, 1, vocab_size)

    model = Model(inputs=[i_story_sentence, i_question], outputs=[answer])

    model.compile(
      optimizer=RMSprop(lr=5e-3),
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy']
    )

    print('Single Supporting Fact model details:')
    print(model.summary())

    debug_model = Model([i_story_sentence, i_question], [hop1_weights, hop2_weights])

    return model, debug_model


def run_model(scenario):
    babi_data = get_babi_qa_data(scenario)

    if scenario == 'single_supporting_fact_10k':
        model, weights_model = build_single_supporting_fact_models(babi_data)
    elif scenario == 'two_supporting_facts_10k':
        model, weights_model = build_two_supporting_facts_models(babi_data)
    else:
        raise Exception(f"Provided scenario '{scenario}' is not supported!")

    fit_model(model, babi_data, desc=f'Memory network for {scenario}')

    for _ in range(3):
        print_weights(babi_data, model, weights_model, scenario)


if __name__ == '__main__':
    # single supporting fact
    BATCH_SIZE = 32
    EPOCHS = 40
    EMBEDDING_DIM = 15
    print('\n\n##################################################\nRunning single supporting fact scenario...')
    run_model('single_supporting_fact_10k')

    # two supporting facts
    BATCH_SIZE = 32
    EPOCHS = 30
    EMBEDDING_DIM = 30
    print('\n\n##################################################\nRunning two supporting facts scenario...')
    run_model('two_supporting_facts_10k')
