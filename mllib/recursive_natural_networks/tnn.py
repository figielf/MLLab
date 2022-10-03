from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


class tnn:
    """
    recursive neural network classifier based on binary ensamble with buildin embedding layer
    tree has to be instance of tests.utils.nlp_data_utils.Tree
    """

    def __init__(self, V, D, K, activation, quadratic_logits=False, params0=None):
        self.V = V
        self.D = D
        self.K = K
        self.activation = activation
        self.quadratic_logits = quadratic_logits

        self._init_params(params0)
        self._fitted_params = None

    def _init_params(self, params0=None):
        if params0 is not None:
            if self.quadratic_logits:
                We, W2hl, W2hr, W2hlr, Whl, Whr, bh, Wo, bo = params0
            else:
                We, Whl, Whr, bh, Wo, bo = params0
        else:
            We = np.random.randn(self.V, self.D) / np.sqrt(self.D + self.K)
            if self.quadratic_logits:
                W2hl = np.random.randn(self.D, self.D, self.D) / np.sqrt(3 * self.D)
                W2hr = np.random.randn(self.D, self.D, self.D) / np.sqrt(3 * self.D)
                W2hlr = np.random.randn(self.D, self.D, self.D) / np.sqrt(3 * self.D)
            Whl = np.random.randn(self.D, self.D) / np.sqrt(self.D + self.D)
            Whr = np.random.randn(self.D, self.D) / np.sqrt(self.D + self.D)
            bh = np.zeros(self.D)
            Wo = np.random.randn(self.D, self.K) / np.sqrt(self.D + self.K)
            bo = np.zeros(self.K)

        # embedding matrix
        self.We = tf.Variable(We.astype(np.float32))

        # quadratic term params
        if self.quadratic_logits:
            self.W2hl = tf.Variable(W2hl.astype(np.float32))
            self.W2hr = tf.Variable(W2hr.astype(np.float32))
            self.W2hlr = tf.Variable(W2hlr.astype(np.float32))

        # left and right node params
        self.Whl = tf.Variable(Whl.astype(np.float32))
        self.Whr = tf.Variable(Whr.astype(np.float32))
        self.bh = tf.Variable(bh.astype(np.float32))

        # tree node out params
        self.Wo = tf.Variable(Wo.astype(np.float32))
        self.bo = tf.Variable(bo.astype(np.float32))

    def fit(self, x_trees, learning_rate=1e-1, mu=0.9, reg=0.1, n_epochs=10):
        labels = []
        cost_ops = []
        train_ops = []
        prediction_ops = []
        for tree_root in x_trees:
            node_logits = self.forward_logits_one_sample(tree_root)
            node_labels = self.get_labels_one_sample(tree_root)
            loss_op = self.calc_cost_one_sample(node_logits, node_labels, reg)
            prediction_op = tf.argmax(input=node_logits, axis=1)
            train_op = tf.compat.v1.train.MomentumOptimizer(learning_rate, mu).minimize(loss_op)

            train_ops.append(train_op)
            cost_ops.append(loss_op)
            prediction_ops.append(prediction_op)
            labels.append(node_labels)

        all_sample_costs = []
        with tf.compat.v1.Session() as session:
            init_op = tf.compat.v1.global_variables_initializer()
            session.run(init_op)

            history = {'cost': [],
                       'accuracy': []}
            for epoch in range(n_epochs):
                t0 = datetime.now()
                train_ops_for_epoch, cost_ops_for_epoch, prediction_ops_for_epoch, labels_for_epoch = \
                    shuffle(train_ops, cost_ops, prediction_ops, labels)

                epoch_costs = []
                epoch_correct_predictions = []
                epoch_nodes = []
                for train_op, cost_op, prediction_op, targets \
                        in zip(train_ops_for_epoch, cost_ops_for_epoch, prediction_ops_for_epoch, labels_for_epoch):
                    _, c, y_hat = session.run([train_op, cost_op, prediction_op])

                    epoch_correct_predictions.append(np.sum(y_hat == targets))
                    epoch_nodes.append(len(targets))

                    all_sample_costs.append(c)
                    epoch_costs.append(c)

                accuracy = np.sum(epoch_correct_predictions) / np.sum(epoch_nodes)
                cost = np.mean(epoch_costs)
                print(f'epoch {epoch} - loss: {cost}, accuracy: {accuracy}, elapsed time:{datetime.now() - t0}')
                history['accuracy'].append(accuracy)
                history['cost'].append(cost)

            if self.quadratic_logits:
                self._fitted_params = session.run([self.We, self.W2hl, self.W2hr, self.W2hlr, self.Whl, self.Whr, self.bh, self.Wo, self.bo])
            else:
                self._fitted_params = session.run([self.We, self.Whl, self.Whr, self.bh, self.Wo, self.bo])

        return history, all_sample_costs

    def _tensor_mul(self, d, x1, A, x2):
        A = tf.reshape(A, [d, d*d])
        # (1 x d) x (d x dd)
        tmp = tf.matmul(x1, A)
        # (1 x dd)
        tmp = tf.reshape(tmp, [d, d])
        # (d x d)
        tmp = tf.matmul(tmp, tf.transpose(x2))
        # (d x 1)
        return tf.reshape(tmp, [1, d])

    def get_labels_one_sample(self, tree):
        # recursive post-order tree traversal
        if tree is None:
            return []
        return self.get_labels_one_sample(tree.left) + self.get_labels_one_sample(tree.right) + [tree.label]

    def forward_logits_one_sample(self, tree):
        logits = []
        if self.quadratic_logits:
            self.forward_quadratic_logits_recursive_one_sample(tree, logits)
        else:
            self.forward_logits_recursive_one_sample(tree, logits)
        return tf.concat(logits, axis=0)

    def forward_quadratic_logits_recursive_one_sample(self, tree, logits):
        # recursive post-order tree traversal
        if tree.word is not None:
            x = tf.nn.embedding_lookup(params=self.We, ids=[tree.word])
            logits_left, logits_right = [], []
        else:
            h_left = self.forward_quadratic_logits_recursive_one_sample(tree.left, logits)
            h_right = self.forward_quadratic_logits_recursive_one_sample(tree.right, logits)
            x_logits = self._tensor_mul(self.D, h_left, self.W2hl, h_left) +\
                       self._tensor_mul(self.D, h_right, self.W2hr, h_right) +\
                       self._tensor_mul(self.D, h_left, self.W2hlr, h_right) +\
                       tf.matmul(h_left, self.Whl) +\
                       tf.matmul(h_right, self.Whr) +\
                       self.bh
            x = self.activation(x_logits)

        node_output_logits = tf.matmul(x, self.Wo) + self.bo
        logits.append(node_output_logits)
        return x

    def forward_logits_recursive_one_sample(self, tree, logits):
        # recursive post-order tree traversal
        if tree.word is not None:
            x = tf.nn.embedding_lookup(params=self.We, ids=[tree.word])
            logits_left, logits_right = [], []
        else:
            h_left = self.forward_logits_recursive_one_sample(tree.left, logits)
            h_right = self.forward_logits_recursive_one_sample(tree.right, logits)
            x_logits = tf.matmul(h_left, self.Whl) + tf.matmul(h_right, self.Whr) + self.bh
            x = self.activation(x_logits)

        node_output_logits = tf.matmul(x, self.Wo) + self.bo
        logits.append(node_output_logits)
        return x

    def calc_cost_one_sample(self, logits, labels, reg):
        nodes_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels,
            logits=logits
        )

        all_params = [self.We, self.Whl, self.Whr, self.Wo]  # , self.bh, self.bo]
        cost = tf.reduce_mean(nodes_entropy) + reg * sum([tf.nn.l2_loss(p) for p in all_params])
        return cost

    def score(self, x_trees, root_only=False, reg=0.1):
        model = tnn(self.V, self.D, self.K, self.activation, self._fitted_params)

        labels = []
        cost_ops = []
        prediction_ops = []
        for tree_root in x_trees:
            node_logits = model.forward_logits_one_sample(tree_root)
            node_labels = model.get_labels_one_sample(tree_root)
            loss_op = model.calc_cost_one_sample(node_logits, node_labels, reg)
            prediction_op = tf.argmax(input=node_logits, axis=1)

            cost_ops.append(loss_op)
            prediction_ops.append(prediction_op)
            labels.append(node_labels)

        with tf.compat.v1.Session() as session:
            init_op = tf.compat.v1.global_variables_initializer()
            session.run(init_op)

            correct_predictions_all_nodes = []
            correct_predictions_root = []
            nodes = []
            costs = []
            for cost_op, prediction_op, targets in zip(cost_ops, prediction_ops, labels):
                c, y_hat = session.run([cost_op, prediction_op])

                correct_predictions_all_nodes.append(np.sum(y_hat == targets))
                correct_predictions_root.append(y_hat[-1] == targets[-1])
                nodes.append(len(targets))
                costs.append(c)
            if root_only:
                accuracy = np.sum(correct_predictions_root) / len(x_trees)
            else:
                accuracy = np.sum(correct_predictions_all_nodes) / np.sum(nodes)
            total_cost = np.mean(costs)

        return accuracy, total_cost
