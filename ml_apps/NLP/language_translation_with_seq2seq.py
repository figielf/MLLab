import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Embedding, Input, LSTM, GlobalMaxPool1D, Bidirectional, Flatten

from NLP.glove_utils import load_glove_embeddings, get_embedding_matrix
from NLP.nlp_data_utils import get_spanish_english_translations_data


def plot_history(history):
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 2, 1)
    plt.plot(history['loss'], label='loss')
    plt.plot(history['val_loss'], label='val_loss')
    plt.legend()

    # accuracies
    plt.subplot(1, 2, 2)
    plt.plot(history['custom_accuracy'], label='acc')
    plt.plot(history['val_custom_accuracy'], label='val_acc')
    plt.legend()
    plt.show()


def fit_model(model, x, y, desc):
    print(f'Training {desc} model...')
    r = model.fit(
        x,
        y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=0.2,
    )
    plot_history(r.history)


def build_models(encoder_vocab_size, decoder_vocab_size, encoder_seq_len, decoder_seq_len, encoder_emb_init):
    # define train model with teacher forcing
    emb_encoder = Embedding(
        input_dim=encoder_vocab_size,
        output_dim=EMBEDDING_DIM,
        weights=[encoder_emb_init],
        input_length=encoder_seq_len,
        # trainable=False,
        name='encoder_embedding'
    )
    lstm_encoder = LSTM(LATENT_DIM, return_state=True, name='encoder_lstm')
    emb_decoder = Embedding(
        input_dim=decoder_vocab_size,
        output_dim=EMBEDDING_DIM,
        trainable=True,
        name='decoder_embedding'
    )

    lstm_decoder = LSTM(LATENT_DIM, return_state=True, return_sequences=True, name='decoder_lstm')
    dense_decoder = Dense(decoder_vocab_size, activation='softmax', name='decoder_dense')
    ie = Input(shape=(encoder_seq_len,), name='encoder_input')  # output shape=(BATCH_SIZE, max_input_len)
    xe = emb_encoder(ie)  # output shape=(BATCH_SIZE, max_input_len, EMBEDDING_DIM)
    _, he, ce = lstm_encoder(xe)  # output shape=(BATCH_SIZE, LATENT_DIM)

    id = Input(shape=(decoder_seq_len,), name='decoder_input')  # output shape=(BATCH_SIZE, max_target_len)
    xd = emb_decoder(id)  # output shape=(BATCH_SIZE, max_target_len, EMBEDDING_DIM)
    xd, _, _ = lstm_decoder(xd, initial_state=[he, ce])  # output shape=(BATCH_SIZE, max_target_len, LATENT_DIM)
    output = dense_decoder(xd)  # output shape=(BATCH_SIZE, max_target_len, num_words_output)

    train_model = Model(inputs=[ie, id], outputs=[output])

    def custom_loss(y_true, y_pred):
        y_mask = tf.cast(tf.greater(y_true, 0), dtype='float32')
        out = y_mask * y_true * tf.math.log(y_pred)
        return -tf.math.reduce_sum(out) / tf.reduce_sum(y_mask)

    def custom_accuracy(y_true, y_pred):
        targ = tf.math.argmax(y_true, axis=-1)
        pred = tf.math.argmax(y_pred, axis=-1)
        correct = tf.cast(tf.math.equal(targ, pred), dtype='float32')

        # 0 is padding, don't include those
        mask = tf.cast(tf.greater(targ, 0), dtype='float32')
        n_correct = tf.math.reduce_sum(mask * correct)
        n_total = tf.math.reduce_sum(mask)
        return n_correct / n_total

    #train_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    train_model.compile(loss=custom_loss, optimizer='adam', metrics=[custom_accuracy])
    print('Training model details:')
    print(train_model.summary())

    encoder_model = Model(inputs=ie, outputs=[he, ce])
    print('Encoder model details:')
    print(encoder_model.summary())

    id_translate = Input(shape=(1,), name='translation_decoder_input')  # output shape=(BATCH_SIZE, 1)
    init_h = Input(shape=(LATENT_DIM), name='translation_init_h')  # output shape=(LATENT_DIM, )
    init_c = Input(shape=(LATENT_DIM), name='translation_init_c')  # output shape=(LATENT_DIM, )
    xd = emb_decoder(id_translate)  # output shape=(BATCH_SIZE, max_target_len, EMBEDDING_DIM)
    xd, hd, cd = lstm_decoder(xd, initial_state=[init_h, init_c])  # output shape=(BATCH_SIZE, max_target_len, LATENT_DIM)
    output = dense_decoder(xd)  # output shape=(BATCH_SIZE, max_target_len, num_words_output)
    decoder_model = Model(inputs=[id_translate, init_h, init_c], outputs=[output, hd, cd])
    print('Decoder model details:')
    print(decoder_model.summary())

    # Save model
    train_model.save('translations_seq2seq.h5')
    print('Train model saved in translations_seq2seq.h5')
    return train_model, encoder_model, decoder_model


def translate(x, encoder, decoder, decoder_word2idx, decoder_idx2word, decoder_max_line_length):
    h, c = encoder.predict(x)

    end_idx = decoder_word2idx['<eos>']
    next_word = np.array([[decoder_word2idx['<sos>']]])

    translation = []
    for _ in range(decoder_max_line_length):
        y_hat, h, c = decoder.predict([next_word, h, c])
        predictions = y_hat[0, 0, :]
        predictions[0] = 0
        word_idx = np.argmax(predictions)
        if word_idx == end_idx:
            break
        translation.append(decoder_idx2word[word_idx])
        next_word = np.array([[word_idx]])
    return translation


if __name__ == '__main__':
    BATCH_SIZE = 64  # Batch size for training.
    EPOCHS = 40  # Number of epochs to train for.
    LATENT_DIM = 256  # Latent dimensionality of the encoding space.
    NUM_SAMPLES = 10000  # Number of samples to train on.
    MAX_NUM_WORDS = 20000
    EMBEDDING_DIM = 100
    glove_folder = 'C:/dev/my_private/MLLab/temp_files/glove.6B'

    input_texts, target_texts, target_texts_inputs = get_spanish_english_translations_data(n_samples=NUM_SAMPLES)

    input_tokenizer = Tokenizer(MAX_NUM_WORDS)
    input_tokenizer.fit_on_texts(input_texts)
    input_word2idx = input_tokenizer.word_index
    print(f'Found {len(input_word2idx)} unique input tokens.')
    input_idx2word = {v: k for k, v in input_word2idx.items()}
    input_seq = input_tokenizer.texts_to_sequences(input_texts)
    max_input_len = max(len(s) for s in input_seq)
    print('max input sequence len:', max_input_len)
    encoder_inputs = pad_sequences(input_seq, maxlen=max_input_len)

    target_tokenizer = Tokenizer(MAX_NUM_WORDS, filters='')
    target_tokenizer.fit_on_texts(target_texts + target_texts_inputs)
    target_word2idx = target_tokenizer.word_index
    print(f'Found {len(target_word2idx)} unique target tokens.')
    assert('<sos>' in target_word2idx)
    assert('<eos>' in target_word2idx)
    target_idx2word = {v: k for k, v in target_word2idx.items()}
    target_seq = target_tokenizer.texts_to_sequences(target_texts)
    target_seq_inputs = target_tokenizer.texts_to_sequences(target_texts_inputs)
    max_target_len = max(len(s) for s in target_seq)
    print('max target sequence len:', max_target_len)
    decoder_inputs = pad_sequences(target_seq_inputs, maxlen=max_target_len, padding='post')
    decoder_targets = pad_sequences(target_seq, maxlen=max_target_len, padding='post')
    num_words_output = len(target_word2idx) + 1

    word2glovevec = load_glove_embeddings(glove_folder, EMBEDDING_DIM)
    num_words = min(MAX_NUM_WORDS, len(input_word2idx) + 1)
    glove_embedding_matrix = get_embedding_matrix(num_words, word2glovevec, input_word2idx)

    # one hot targets
    decoder_targets_one_hot = np.zeros((len(decoder_targets), max_target_len, num_words_output), dtype='float32')
    for i, target_sequence in enumerate(decoder_targets):
        for t, word in enumerate(target_sequence):
            if word > 0:
                decoder_targets_one_hot[i, t, word] = 1

    train_model, encoder_model, decoder_model = build_models(num_words, num_words_output, max_input_len, max_target_len, glove_embedding_matrix)
    fit_model(train_model, x=[encoder_inputs, decoder_inputs], y=decoder_targets_one_hot, desc='LSTM')

    random_translations = np.random.choice(NUM_SAMPLES, size=10, replace=False)
    for id in random_translations:
        txt = encoder_inputs[id:id+1]
        translated = translate(txt, encoder_model, decoder_model, target_word2idx, target_idx2word, max_target_len)
        print('oryginal:\n', input_texts[id:id+1])
        print('predicted translation:', '\n', ' '.join(translated), '\n')
        print('actual translation:', '\n', target_texts[id], '\n')
