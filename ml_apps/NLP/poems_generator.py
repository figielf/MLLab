import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Embedding, Input, LSTM, GlobalMaxPool1D, Bidirectional, Flatten
from tensorflow.keras.optimizers import Adam

from NLP.glove_utils import load_glove_embeddings, get_embedding_matrix, get_robert_frost_data_for_seq2seq

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

def plot_history(history):
    plt.figure(figsize=(24, 10))
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

def run_model(model, x, y, desc):
    print(f'Training {desc} model...')
    print(model.summary())

    z = np.zeros((len(x), LATENT_DIM))
    r = model.fit(
        [x, z, z],
        y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT
    )

    plot_history(r.history)


def sample_poem_line(model, word2idx, idx2word, max_line_length):
    line_start = np.array([[word2idx['<sos>']]])
    h = np.zeros((1, LATENT_DIM))
    c = np.zeros((1, LATENT_DIM))

    end_idx = word2idx['<eos>']

    poem = []
    for _ in range(max_line_length):
        y_hat, h, c = model.predict([line_start, h, c])
        word_distrib = y_hat[0, 0, :] / y_hat[0, 0, :].sum()
        word_idx = np.random.choice(len(word_distrib), size=1, replace=True, p=word_distrib)
        word_idx = word_idx[0]
        if word_idx == end_idx:
            break
        word = idx2word[word_idx]
        poem.append(word)
        line_start = np.array([[word_idx]])
    return poem


if __name__ == '__main__':
    MAX_SEQUENCE_LENGTH = 100
    MAX_VOCAB_SIZE = 3000
    EMBEDDING_DIM = 50
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 128
    EPOCHS = 2000
    LATENT_DIM = 32
    glove_folder = 'C:/dev/my_private/MLLab/temp_files/glove.6B'

    input_texts, target_texts = get_robert_frost_data_for_seq2seq()

    tokenizer = Tokenizer(MAX_VOCAB_SIZE, filters='')
    tokenizer.fit_on_texts(input_texts + target_texts)
    word2idx = tokenizer.word_index
    V = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
    print(f'Found {V} unique tokens.')
    assert('<sos>' in word2idx)
    assert('<eos>' in word2idx)
    idx2word = {v:k for k, v in word2idx.items()}

    input_seq = tokenizer.texts_to_sequences(input_texts)
    target_seq = tokenizer.texts_to_sequences(target_texts)
    max_seq_length = max(len(s) for s in input_seq)
    print('Max sentence length:', max_seq_length)
    sequence_length = min(max_seq_length, MAX_SEQUENCE_LENGTH)
    input_seq_padded = pad_sequences(input_seq, maxlen=sequence_length, padding='post')
    target_seq_padded = pad_sequences(target_seq, maxlen=sequence_length, padding='post')
    print('input_seq_padded.shape:', input_seq_padded.shape)
    print('target_seq_padded.shape:', target_seq_padded.shape)

    word2glovevec = load_glove_embeddings(glove_folder, EMBEDDING_DIM)
    glove_embedding_matrix = get_embedding_matrix(V, word2glovevec, word2idx)

    # define layers to share between encoder and decoder models
    emb1 = Embedding(
        input_dim=V,
        output_dim=EMBEDDING_DIM,
        weights=[glove_embedding_matrix],
        #trainable=False
    )
    lstm1 = LSTM(LATENT_DIM, return_sequences=True, return_state=True)
    dense1 = Dense(V, activation='softmax')

    # encoder model
    encoder_input = Input(shape=(sequence_length,))
    init_h = Input(shape=(LATENT_DIM))
    init_c = Input(shape=(LATENT_DIM))
    # output = dense1(lstm1(emb1(encoder_input)))
    x = emb1(encoder_input)
    x, _, _ = lstm1(x, initial_state=[init_h, init_c])
    encoder_output = dense1(x)


    encoder_model = Model([encoder_input, init_h, init_c], encoder_output)
    encoder_model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.01), metrics=['accuracy'])

    one_hot_targets = np.zeros((len(input_seq_padded), sequence_length, V))
    for i, target_sequence in enumerate(target_seq_padded):
        for t, word in enumerate(target_sequence):
            if word > 0:
                one_hot_targets[i, t, word] = 1

    run_model(encoder_model, input_seq_padded, one_hot_targets, 'LSTM')

    # poem generation model (sampling words model)
    decoder_input = Input(shape=(1,))
    # output = dense1(lstm1(emb1(decoder_input)))
    x = emb1(decoder_input)
    x, h, c = lstm1(x, initial_state=[init_h, init_c])
    decoder_output = dense1(x)
    decoder_model = Model([decoder_input, init_h, init_c], [decoder_output, h, c])

    for _ in range(10):
        print('\n')
        for _ in range(4):
            generated_line = sample_poem_line(decoder_model, word2idx, idx2word, max_seq_length)
            print(' '.join(generated_line))
