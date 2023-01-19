import tensorflow as tf

from gans.gan import gan


def get_data():
    (X_train, Y_train), (X_test, Y_test) = tf.keras.datasets.mnist.load_data()

    X_train = X_train * 2 / 255.0 - 1
    x_test = X_test * 2 / 255.0 - 1

    N, H, W = X_train.shape
    D = H * W
    X_train = X_train.reshape(-1, D)
    X_test = x_test.reshape(-1, D)
    return X_train, X_test, Y_train, Y_test, (H, W)


if __name__ == '__main__':
    BATCH_SIZE = 32
    EPOCH = 30000

    X_train, X_test, Y_train, Y_test, picture_shape = get_data()
    D = picture_shape[0] * picture_shape[1]

    model = gan(latent_dim=100, D=D)
    history = model.fit(X_train, EPOCH, BATCH_SIZE, save_images_details={'sample_period': 5000, 'image_shape': picture_shape}, log_period=1000)

    model.plot_history(history)
