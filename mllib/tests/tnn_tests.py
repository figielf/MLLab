import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from recursive_natural_networks.tnn import tnn
from tests.utils.nlp_data_utils import get_trees_data_with_word2idx_from_ptb

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


if __name__ == '__main__':
    train_data, test_data, word2idx = get_trees_data_with_word2idx_from_ptb()

    train_data = train_data[:10]
    test_data = test_data[:10]

    V = len(word2idx)
    D = 80
    K = 5

    # word embedding
    We = init_weight(V, D)
    # linear terms
    W1 = init_weight(D, D)
    W2 = init_weight(D, D)
    # bias
    bh = np.zeros(D)
    # output layer
    Wo = init_weight(D, K)
    bo = np.zeros(K)

    model = tnn(V, D, K, tf.nn.relu, quadratic_logits=False, params0=[We, W1, W2, bh, Wo, bo])
    history, all_costs = model.fit(train_data, learning_rate=1e-1, n_epochs=5)
    train_accuracy, train_total_cost = model.score(train_data, root_only=True)
    test_accuracy, test_total_cost = model.score(test_data, root_only=True)
    print(f'train sentence sentiment accuracy: {train_accuracy}, train cost: {train_total_cost}')
    print(f'test sentence sentiment accuracy: {test_accuracy}, test cost: {test_total_cost}')

    plt.figure(figsize=(16, 16))
    plt.subplot(3, 1, 1)
    plt.plot(history['cost'], label='cost per epoch')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(all_costs, label='cost for each sample')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(history['accuracy'], label='root level accuracy')
    plt.legend()
    plt.show()
