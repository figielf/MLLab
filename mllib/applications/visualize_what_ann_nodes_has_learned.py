import pickle

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt

from activations import sigmoid
from natural_networks.ann import ann_classifier
from tests.ann_classifier_tests import unwind_history
from tests.utils.data_utils import get_mnist_data

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


class simple_ann_predictor:
    class simple_ann_layer:
        def __init__(self, W, b):
            self.W = W
            self.b = b

        def forward(self, x):
            return sigmoid(x.dot(self.W) + self.b)

        def forward_tf(self, x):
            return tf.nn.sigmoid(tf.matmul(x, self.W) + self.b)

    def __init__(self, ann_persisted_params):
        self.layers = []
        print('Creating simple_ann_predictor model with layers:')
        for i, (W, b) in enumerate(ann_persisted_params):
            self.layers.append(self.simple_ann_layer(W, b))
            print(f'layer {i} in-out shape: {W.shape}')

    def forward(self, x, out_layer=None):
        if out_layer is None:
            out_layer = len(self.layers)
        z = x
        for n_l in range(0, out_layer):
            z = self.layers[n_l].forward(z)
        return z

    def forward_tf(self, x, out_layer=None):
        if out_layer is None:
            out_layer = len(self.layers)
        z = x
        for n_l in range(0, out_layer):
            z = self.layers[n_l].forward_tf(z)
        return z

    def predict(self, x):
        z_hat = self.forward(x)
        print(z_hat)
        return z_hat.argmax(axis=1)

    def get_layer_out_dims(self, layer_number=None):
        model_in_size = self.layers[0].W.shape[0]

        layer_out_size = self.W.shape[1]  \
                            if layer_number is None or layer_number > len(self.layers) \
                            else self.layers[layer_number].W.shape[1]
        return model_in_size, layer_out_size


class ann_param_persistor:
    def save(self, ann_model: ann_classifier, file_path):
        arrays = [(l.W.eval(), l.b.eval()) for l in ann_model.hidden_layers]
        arrays.append((ann_model.W.eval(), ann_model.b.eval()))
        with open(file_path, 'wb') as f:
            pickle.dump(arrays, f)

    def load(self, file_path) -> simple_ann_predictor:
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
        return simple_ann_predictor(params)


class logistic_layered_model_visualizer:
    def __init__(self, model):
        self.model = model

    def show_layer(self, node_to_visualize, n_epochs):
        layer, node = node_to_visualize
        assert 0 <= layer < len(self.model.layers)
        assert 0 <= node
        D = self.model.layers[0].W.shape[0]
        M_1, M = self.model.layers[layer].W.shape \
                 if layer < len(self.model.layers) \
                 else self.model.W.shape
        assert node <= M

        ones = np.zeros((M, 1))
        ones[node, 0] = 1

        x_init = np.random.randn(1, D) / np.sqrt(D)
        x = tf.Variable(x_init.astype(np.float32))
        z_hat = self.model.forward_tf(x, layer + 1)
        nodes_softmax = tf.nn.softmax(z_hat)
        chosen_node_softmax = tf.matmul(nodes_softmax, ones)
        cost = tf.reshape(-chosen_node_softmax, shape=(1,))
        train_op = tf.compat.v1.train.AdamOptimizer(0.01).minimize(cost)

        history = []
        with tf.compat.v1.Session() as local_session:
            local_session.run(tf.compat.v1.global_variables_initializer())
            for e in range(n_epochs):
                _, c = local_session.run((train_op, cost))
                history.append(c)
                if e % 1000 == 0:
                    print(f'generate image of node - epoch: {e}, cost: {c}')
            result = x.eval()
        return result, history

def train_ann_and_persist_params(Xtrain, Ytrain, Xtest, Ytest, params_file_path):
    model = ann_classifier([1000, 750, 500])
    with tf.compat.v1.Session() as session:
        model.set_session(session)
        history = model.fit(Xtrain.copy(), Ytrain.copy(), Xtest.copy(), Ytest.copy(), n_epochs=50, batch_size=1000, learning_rate=1e-3)
        persistor = ann_param_persistor()
        persistor.save(model, params_file_path)
        print('ann_classifier model params saved to', params_file_path)

    train_cost, test_cost, train_error, test_error, params = unwind_history(history, n_layers=4)
    plt.figure(figsize=(16, 16))
    plt.subplot(2, 1, 1)
    plt.plot(train_cost, label='train cost')
    plt.plot(test_cost, label='test cost')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(train_error, label='train error')
    plt.plot(test_error, label='test error')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    K = 10
    test_size = 1000
    Xtrain, _, Ytrain, _, picture_shape = get_mnist_data(train_size=1, should_plot_examples=False)
    Xtrain, Ytrain = Xtrain[:-test_size], Ytrain[:-test_size]
    Xtest, Ytest = Xtrain[-test_size:], Ytrain[-test_size:]

    params_file_path = 'c://temp//ann_params_3layers.bin'
    #train_ann_and_persist_params(Xtrain.copy(), Ytrain.copy(), Xtest.copy(), Ytest.copy(), params_file_path)

    persistor = ann_param_persistor()
    loaded_model = persistor.load(params_file_path)
    print('ann_classifier model loaded from', params_file_path)

    node = (3, 3)
    visualizer = logistic_layered_model_visualizer(loaded_model)
    result, visualizer_history = visualizer.show_layer(node, n_epochs=100000)

    plt.subplot(2, 2, 1)
    plt.subplot(2, 1, 1)
    plt.imshow(result.reshape(picture_shape), cmap='gray')
    plt.title(f'See what node {node[0]} from layer {node[1]} has learned')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(visualizer_history, label='visualizer train cost')
    plt.show()
