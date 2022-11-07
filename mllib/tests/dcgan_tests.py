import tensorflow as tf

from gans.dcgan import dcgan_tf1
from tests.utils.data_utils import get_mnist_data

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


if __name__ == '__main__':
    K = 10
    N = 1000
    X, _, Y, _, picture_shape = get_mnist_data(train_size=1.0, should_plot_examples=False)
    X = X.reshape(len(X), 28, 28, 1)
    print('X.shape:', X.shape)
    dim = X.shape[1]
    colors = X.shape[-1]

    d_sizes = {
        'conv_layers': [(2, 5, 2, False), (64, 5, 2, True)],
        'dense_layers': [(1024, True)],
    }
    g_sizes = {
        'z': 100,
        'projection': 128,
        'bn_after_project': False,
        'conv_layers': [(128, 5, 2, True), (colors, 5, 2, False)],
        'dense_layers': [(1024, True)],
        'output_activation': tf.sigmoid,
    }

    gan = dcgan_tf1(dim, colors, d_sizes, g_sizes)
    gan.fit(X)
