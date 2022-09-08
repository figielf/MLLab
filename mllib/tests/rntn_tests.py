import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from recursive_natural_networks.rntn import rntn
from tests.utils.nlp_data_utils import get_trees_data_with_word2idx_from_ptb

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


def init_weight(Mi, Mo):
    return np.random.randn(Mi, Mo) / np.sqrt(Mi + Mo)


def label2binary(label):
    if label > 2:
        return 1
    elif label < 2:
        return 0
    else:
        return -1


def filter_out_binary(data):
    return [d for d in data if d[3][-1] >= 0]


def init_params_common(V, D, K):
    We = init_weight(V, D)
    W11 = np.random.randn(D, D, D) / np.sqrt(3 * D)
    W22 = np.random.randn(D, D, D) / np.sqrt(3 * D)
    W12 = np.random.randn(D, D, D) / np.sqrt(3 * D)
    W1 = init_weight(D, D)
    W2 = init_weight(D, D)
    bh = np.zeros(D)
    Wo = init_weight(D, K)
    bo = np.zeros(K)

    return We, W11, W22, W12, W1, W2, bh, Wo, bo


if __name__ == '__main__':
    is_binary = True
    train_data_org, test_data_org, word2idx_org = get_trees_data_with_word2idx_from_ptb()

    train_data_org = train_data_org[:100]
    test_data_org = test_data_org[:100]

    binary_label_modifier = label2binary if is_binary else None

    train_data, test_data, word2idx = get_trees_data_with_word2idx_from_ptb()
    V = len(word2idx)
    print("vocab size:", V)
    D = 10
    K = 2 if is_binary else 5

    params0 = init_params_common(V, D, K)
    model = rntn(V, D, K, params0=params0)
    train_data_processed = []
    for data in train_data:
        train_data_processed.append(model.preprocess_tree_data_for_model(data, label_modifier=binary_label_modifier))
    test_data_processed = []
    for data in test_data:
        test_data_processed.append(model.preprocess_tree_data_for_model(data, label_modifier=binary_label_modifier))

    if is_binary:
        train_data_processed = filter_out_binary(train_data_processed)
        test_data_processed = filter_out_binary(test_data_processed)

    history = model.fit(train_data_processed, test_data_processed[:100], reg=1e-3, n_epochs=2, train_inner_nodes=False)

    plt.figure(figsize=(16, 16))
    plt.subplot(2, 1, 1)
    plt.plot(history['train_cost'], label='cost per epoch')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history['train_accuracy'], label='cost for each sample')
    plt.plot(history['test_accuracy'], label='root level accuracy')
    plt.legend()
    plt.show()
