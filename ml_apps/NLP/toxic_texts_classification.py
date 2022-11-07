import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Embedding, Input, LSTM, \
    GlobalMaxPool1D, Bidirectional

from NLP.glove_utils import load_glove_embeddings, get_embedding_matrix
from NLP.nlp_data_utils import get_toxic_data

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
    r = model.fit(
        x,
        y,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT
    )

    print(model.summary())

    plot_history(r.history)

    # plot the mean AUC over each label
    p = model.predict(x)
    aucs = []
    for j in range(6):
        auc = roc_auc_score(y[:, j], p[:, j])
        aucs.append(auc)
    print(np.mean(aucs))


if __name__ == '__main__':
    MAX_SEQUENCE_LENGTH = 100
    MAX_VOCAB_SIZE = 20000
    EMBEDDING_DIM = 50
    VALIDATION_SPLIT = 0.2
    BATCH_SIZE = 128
    EPOCHS = 10
    glove_folder = 'C:/dev/my_private/MLLab/temp_files/glove.6B'

    Xtrain, _, Ytrain, _, label_names = get_toxic_data(train_size=1)

    # Xtrain = Xtrain[:1000]
    # Ytrain = Ytrain[:1000]

    tokenizer = Tokenizer(MAX_VOCAB_SIZE)
    tokenizer.fit_on_texts(Xtrain)
    word2idx = tokenizer.word_index
    V = min(MAX_VOCAB_SIZE, len(word2idx) + 1)
    print(f'Found {V} unique tokens.')

    x_seq = tokenizer.texts_to_sequences(Xtrain)
    x_seq_padded = pad_sequences(x_seq, maxlen=MAX_SEQUENCE_LENGTH)

    word2glovevec = load_glove_embeddings(glove_folder, EMBEDDING_DIM)

    glove_embedding_matrix = get_embedding_matrix(V, word2glovevec, word2idx)

    embeddings = Embedding(
        input_dim=V,
        output_dim=EMBEDDING_DIM,
        weights=[glove_embedding_matrix],
        # input_length=MAX_SEQUENCE_LENGTH,
        trainable=False
    )

    # CNN
    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embeddings(input_)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = MaxPooling1D(3)(x)
    x = Conv1D(128, 3, activation='relu')(x)
    x = GlobalMaxPooling1D()(x)
    x = Dense(128, activation='relu')(x)
    output = Dense(len(label_names), activation='sigmoid')(x)

    model = Model(input_, output)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    run_model(model, x_seq_padded, Ytrain, 'CNN')

    # LSTM
    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embeddings(input_)
    x = LSTM(15, return_sequences=True)(x)
    x = GlobalMaxPooling1D()(x)
    output = Dense(len(label_names), activation='sigmoid')(x)

    model = Model(input_, output)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    run_model(model, x_seq_padded, Ytrain, 'LSTM')

    # Simple RNN
    input_ = Input(shape=(MAX_SEQUENCE_LENGTH,))
    x = embeddings(input_)
    x = Bidirectional(LSTM(15, return_sequences=True))(x)
    x = GlobalMaxPooling1D()(x)
    output = Dense(len(label_names), activation='sigmoid')(x)

    model = Model(input_, output)
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    run_model(model, x_seq_padded, Ytrain, 'Bidirectional LSTM')
