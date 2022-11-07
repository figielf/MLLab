import numpy as np
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalMaxPooling1D, Dense, Input, LSTM, Bidirectional, Concatenate, Lambda, Permute

from CV.cv_data_utils import get_mnist_data


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


if __name__ == '__main__':
    D = 7
    T = 28
    M = 15
    BATCH_SIZE = 32
    EPOCHS = 10
    VALIDATION_SPLIT = 0.3

    X_train, _, Y_train, _, picture_shape = get_mnist_data(train_size=1, should_plot_examples=False)
    X_train = X_train.reshape(-1, T, D)
    print(X_train.shape)
    #X_train_flipped = np.transpose(X_train, axes=(0, 2, 1))
    #print(X_train_flipped.shape)

    input_ = Input(shape=(T, D))  # shape = (T, D)
    x = Bidirectional(LSTM(M, return_sequences=True))(input_)  # shape = (T, 2M)
    x = GlobalMaxPooling1D()(x)  # shape = (2M, )
#
    #i_t = Input(shape=(D, T))  # shape = (D, T)
    x_t = Permute((2, 1))(input_)  # shape = (T, D)
    #x_t = Lambda(lambda x: tf.transpose(x, perm=[0, 2, 1]))(input_)  # shape = (T, D)
    x_t = Bidirectional(LSTM(M, return_sequences=True))(x_t)  # shape = (T, 2M)
    x_t = GlobalMaxPooling1D()(x_t)  # shape = (2M, )

    x_out = Concatenate()([x, x_t])  # shape = (4M, )
    output = Dense(10, activation='softmax')(x_out)  # shape = (10, )

    model = Model(inputs=input_, outputs=output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics='accuracy')

    print(model.summary())

    history = model.fit(
        #(X_train, X_train_flipped),
        X_train,
        Y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_split=VALIDATION_SPLIT)

    plot_history(history.history)
