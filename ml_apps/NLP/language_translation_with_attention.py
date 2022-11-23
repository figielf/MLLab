import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, LSTM, GRU, Dense, Embedding, Bidirectional, RepeatVector, Concatenate, Activation, Dot, Lambda, Permute, Reshape

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
    # Tx = encoder_seq_len
    # Ty = decoder_seq_len
    # Me = LATENT_DIM
    # Md = LATENT_DIM_DECODER

    def encode(input_):
        # define train model with teacher forcing
        emb_encoder = Embedding(
            input_dim=encoder_vocab_size,
            output_dim=EMBEDDING_DIM,
            weights=[encoder_emb_init],
            input_length=encoder_seq_len,
            # trainable=False,
            name='encoder_embedding'
        )
        #lstm_encoder = Bidirectional(LSTM(LATENT_DIM, return_sequences=True, name='encoder_lstm'))

        x = emb_encoder(input_)  # output shape=(BATCH_SIZE, Tx, EMBEDDING_DIM)
        h = Bidirectional(LSTM(LATENT_DIM, return_sequences=True, name='encoder_lstm'))(x)  # output shape=(BATCH_SIZE, Tx, 2 x Me)
        return h

    emb_decoder = Embedding(
        input_dim=decoder_vocab_size,
        output_dim=EMBEDDING_DIM,
        trainable=True,
        name='decoder_embedding'
    )

    # make sure we do softmax over the time axis, expected shape is N x T x D
    # note: the latest version of Keras allows you to pass in axis arg
    def softmax_over_time(x):
        assert (len(x.shape) > 2)
        e = tf.math.exp(x - tf.math.reduce_max(x, axis=1, keepdims=True))
        s = tf.math.reduce_sum(e, axis=1, keepdims=True)
        return e / s

    repeat_vector = RepeatVector(n=encoder_seq_len, name='attention_repeat_encoder_h')
    attention_concat = Concatenate(axis=-1, name='attention_s_h_concat')
    attention_dense1 = Dense(encoder_seq_len, activation = 'tanh', name='attention_alpha_nn1')
    attention_dense2 = Dense(1, activation=softmax_over_time, name='attention_alpha_nn2')
    attention_dot = Dot(axes=1)

    def context_by_attention(h, s_prev):
        s_repeated = repeat_vector(s_prev)  # output shape=(BATCH_SIZE, Tx, Md)
        concat = attention_concat([h, s_repeated])  # output shape=(BATCH_SIZE, Tx, Md + 2 x Me)
        alpha = attention_dense1(concat)  # output shape=(BATCH_SIZE, Tx, 10)
        alpha = attention_dense2(alpha)  # output shape=(BATCH_SIZE, Tx, 1)
        context = attention_dot([alpha, h])  # output shape=(BATCH_SIZE, 2 x Me)  # = sum over alpha[t] * h[t]
        alpha_reshape = Reshape((encoder_seq_len, ))(alpha)  # output shape=(BATCH_SIZE, Tx)
        return context, alpha_reshape


    # encoder
    i_encoder = Input(shape=(encoder_seq_len,), name='encoder_input')  # output shape=(BATCH_SIZE, Tx)
    encoder_outputs = encode(i_encoder)  # output shape=(BATCH_SIZE, Tx, 2 x Me)

    # decoder
    i_decoder = Input(shape=(decoder_seq_len, ), name='decoder_input')  # output shape=(BATCH_SIZE, Ty)
    xd = emb_decoder(i_decoder)  # output shape=(BATCH_SIZE, Ty, EMBEDDING_DIM)
    init_s = Input(shape=(LATENT_DIM_DECODER), name='translation_init_s')  # output shape=(LATENT_DIM_DECODER, )
    init_c = Input(shape=(LATENT_DIM_DECODER), name='translation_init_c')  # output shape=(LATENT_DIM_DECODER, )

    decoder_context_word_prev_concat = Concatenate(axis=2, name='decoder_context_true_input_concat')
    decoder_lstm = LSTM(LATENT_DIM_DECODER, return_state=True, name='decoder_lstm')
    decoder_dense = Dense(decoder_vocab_size, activation='softmax', name='decoder_final_dense_output')
    s = init_s
    c = init_c
    outputs = []  # output is a list of len(Ty) and each element of shape=(BATCH_SIZE, decoder_vocab_size)
    for t in range(decoder_seq_len):  # repeat Ty times for every decoder seq step hidden state
        # calc attention for time t
        context, _ = context_by_attention(encoder_outputs, s)  # output shape=(BATCH_SIZE, 2 x Me)

        selector = Lambda(lambda x: x[:, t:t + 1])
        xt = selector(xd)  # output shape=(BATCH_SIZE, 1)

        lstm_input = decoder_context_word_prev_concat([context, xt])  # output shape=(BATCH_SIZE, 2 x Me + EMBEDDING_DIM)

        output, s, c = decoder_lstm(lstm_input, initial_state=[s, c])  # output shape=[(BATCH_SIZE, LATENT_DIM_DECODER), (BATCH_SIZE, LATENT_DIM_DECODER), (BATCH_SIZE, LATENT_DIM_DECODER)]
        output = decoder_dense(output)  # output shape=(BATCH_SIZE, decoder_vocab_size)
        outputs.append(output)

    def stack_and_transpose(x):
        x = tf.stack(x)  # output shape=(Ty, BATCH_SIZE, decoder_vocab_size)
        x = tf.transpose(x, perm=[1, 0, 2])  # output shape=(BATCH_SIZE, Ty, decoder_vocab_size)
        return x

    # make it a layer
    stacker = Lambda(stack_and_transpose)
    outputs = stacker(outputs)  # output shape=(BATCH_SIZE, Ty, decoder_vocab_size)

    train_model = Model(inputs=[i_encoder, i_decoder, init_s, init_c], outputs=outputs)

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

    # prediction models
    encoder_model = Model(inputs=i_encoder, outputs=encoder_outputs)
    print('Encoder model details:')
    print(encoder_model.summary())

    i_translate_decoder = Input(shape=(encoder_seq_len, 2 * LATENT_DIM), name='translation_encoder_input')  # output shape=(BATCH_SIZE, Tx, 2 x Me)
    i_translate_1_word = Input(shape=(1,), name='translation_decoder_input')  # output shape=(BATCH_SIZE, 1)
    translate_encoder_outputs = emb_decoder(i_translate_1_word)  # output shape=(BATCH_SIZE, 1, EMBEDDING_DIM)
    translate_context, alpha = context_by_attention(i_translate_decoder, init_s)  # output shape=(BATCH_SIZE, 2 x Me)
    translate_lstm_input = decoder_context_word_prev_concat([translate_context, translate_encoder_outputs])  # output shape=(BATCH_SIZE, 2 x Me + EMBEDDING_DIM)

    t_output, t_s, t_c = decoder_lstm(translate_lstm_input, initial_state=[init_s, init_c])  # output shape=[(BATCH_SIZE, LATENT_DIM_DECODER), (BATCH_SIZE, LATENT_DIM_DECODER), (BATCH_SIZE, LATENT_DIM_DECODER)]
    t_output = decoder_dense(t_output)  # output shape=(BATCH_SIZE, decoder_vocab_size)

    decoder_model = Model(inputs=[i_translate_1_word, i_translate_decoder, init_s, init_c], outputs=[t_output, t_s, t_c, alpha])
    print('Decoder model details:')
    print(decoder_model.summary())

    # Save model
    train_model.save('translations_attention.h5')
    print('Train model saved in translations_attention.h5')

    return train_model, encoder_model, decoder_model


def translate(x, encoder, decoder, decoder_word2idx, decoder_idx2word, decoder_max_line_length):
    h = encoder.predict(x)

    end_idx = decoder_word2idx['<eos>']
    next_word = np.array([[decoder_word2idx['<sos>']]])
    s = np.zeros((1, LATENT_DIM_DECODER))
    c = np.zeros((1, LATENT_DIM_DECODER))
    translation = []
    alphas = []
    for _ in range(decoder_max_line_length):
        y_hat, s, c, alpha = decoder.predict([next_word, h, s, c])
        alphas.append(alpha[0])
        predictions = y_hat[0, :]
        predictions[0] = 0
        word_idx = np.argmax(predictions)
        if word_idx == end_idx:
            break
        translation.append(decoder_idx2word[word_idx])
        next_word = np.array([[word_idx]])
    return translation, alphas


if __name__ == '__main__':
    BATCH_SIZE = 64
    EPOCHS = 3
    LATENT_DIM = 400
    LATENT_DIM_DECODER = 300  # idea: make it different to ensure things all fit together properly!
    NUM_SAMPLES = 10000
    MAX_SEQUENCE_LENGTH = 90
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


    z = np.zeros((len(encoder_inputs), LATENT_DIM_DECODER))  # initial [s, c]
    fit_model(train_model, x=[encoder_inputs, decoder_inputs, z, z], y=decoder_targets_one_hot, desc='Attention')

    random_translations = np.random.choice(NUM_SAMPLES, size=10, replace=False)
    for id in random_translations:
        txt = encoder_inputs[id:id+1]
        translated, alphas = translate(txt, encoder_model, decoder_model, target_word2idx, target_idx2word, max_target_len)
        print('oryginal:\n', input_texts[id:id+1])
        print('predicted translation:', '\n', ' '.join(translated))
        print('Actual translation:', '\n', target_texts[id])

        print(np.array(alphas).shape)
        print(alphas)
        print(alphas[0])
        print(np.array(alphas[0]).shape)
        print(len(input_texts[id:id+1]))
        print(len(translated))
        print(len(target_texts[id]))
        plt.imshow(np.array(alphas), cmap='gray')
        plt.show()

