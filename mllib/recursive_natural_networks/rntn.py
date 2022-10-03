from datetime import datetime

import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


class rntn:
    """
    recursive neural network classifier based on binary ensamble with buildin embedding layer
    tree has to be instance of tests.utils.nlp_data_utils.Tree
    """

    def __init__(self, V, D, K, activation=tf.tanh, quadratic_logits=True, params0=None):
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

        self.weights = [self.We, self.W2hl, self.W2hr, self.W2hlr, self.Whl, self.Whr, self.Wo]

    def fit(self, x_data, test_data, learning_rate=8e-3, reg=1e-3, n_epochs=8, train_inner_nodes=False):
        N = len(x_data)
        words = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='words')
        left_children = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='left_children')
        right_children = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='right_children')
        labels = tf.compat.v1.placeholder(tf.int32, shape=(None,), name='labels')

        def dot1(a, B):
            return tf.tensordot(a, B, axes=[[0], [1]])

        def dot2(B, a):
            return tf.tensordot(B, a, axes=[[1], [0]])

        def recursive_net_transform(hiddens, n):
            h_left = hiddens.read(left_children[n])
            h_right = hiddens.read(right_children[n])
            return self.activation(
                dot1(h_left, dot2(self.W2hl, h_left)) +
                dot1(h_right, dot2(self.W2hr, h_right)) +
                dot1(h_left, dot2(self.W2hlr, h_right)) +
                dot1(h_left, self.Whl) +
                dot1(h_right, self.Whr) +
                self.bh
            )

        def recurrence(hiddens, n):
            w = words[n]
            # any non-word will have index -1

            h_n = tf.cond(
                pred=w >= 0,
                true_fn=lambda: tf.nn.embedding_lookup(params=self.We, ids=w),
                false_fn=lambda: recursive_net_transform(hiddens, n)
            )
            hiddens = hiddens.write(n, h_n)
            n = tf.add(n, 1)
            return hiddens, n

        def condition(hiddens, n):
            # loop should continue while n < len(words)
            return tf.less(n, tf.shape(input=words)[0])

        hiddens = tf.TensorArray(
            tf.float32,
            size=0,
            dynamic_size=True,
            clear_after_read=False,
            infer_shape=False
        )

        hiddens, _ = tf.while_loop(
            cond=condition,
            body=recurrence,
            loop_vars=[hiddens, tf.constant(0)],
            parallel_iterations=1
        )
        h = hiddens.stack()
        logits = tf.matmul(h, self.Wo) + self.bo

        prediction_op = tf.argmax(input=logits, axis=1)
        self.prediction_op = prediction_op

        rcost = reg * sum(tf.nn.l2_loss(p) for p in self.weights)
        if train_inner_nodes:
            # filter out -1s
            labeled_indices = tf.compat.v1.where(labels >= 0)

            cost_op = tf.reduce_mean(
                input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=tf.gather(logits, labeled_indices),
                    labels=tf.gather(labels, labeled_indices),
                )
            ) + rcost
        else:
            cost_op = tf.reduce_mean(
                input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=logits[-1],
                    labels=labels[-1],
                )
            ) + rcost

        train_op = tf.compat.v1.train.AdagradOptimizer(learning_rate=learning_rate).minimize(cost_op)
        # train_op = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).minimize(cost_op)

        with tf.compat.v1.Session() as session:
            init_op = tf.compat.v1.global_variables_initializer()
            session.run(init_op)

            sequence_indexes = range(N)
            history = {'train_cost': [],
                       'train_accuracy': [],
                       'test_accuracy': []}
            for epoch in range(n_epochs):
                t0 = datetime.now()
                sequence_indexes = shuffle(sequence_indexes)
                x_data = shuffle(x_data)

                epoch_costs = []
                epoch_correct_predictions = []
                epoch_nodes = 0
                for i in sequence_indexes:
                    x_words, x_left_childs, x_right_childs, x_labels = x_data[i]

                    c, y_hat, _ = session.run(
                        (cost_op, prediction_op, train_op),
                        feed_dict={
                            words: x_words,
                            left_children: x_left_childs,
                            right_children: x_right_childs,
                            labels: x_labels
                        }
                    )
                    epoch_costs.append(c)
                    epoch_correct_predictions.append(np.sum(y_hat[-1] == x_labels[-1]))
                    epoch_nodes += 1

                # calculate the test score
                test_epoch_correct_predictions = []
                test_epoch_nodes = 0
                for x_words, x_left_childs, x_right_childs, x_labels in test_data:
                    y_hat = session.run(prediction_op, feed_dict={
                            words: x_words,
                            left_children: x_left_childs,
                            right_children: x_right_childs,
                            labels: x_labels
                    })
                    test_epoch_correct_predictions.append(np.sum(y_hat[-1] == x_labels[-1]))
                    test_epoch_nodes += 1

                accuracy = np.sum(epoch_correct_predictions) / epoch_nodes
                test_accuracy = np.sum(test_epoch_correct_predictions) / test_epoch_nodes
                cost = np.mean(epoch_costs)
                print(f'epoch {epoch} - loss: {cost}, accuracy: {accuracy}, test accuracy:{test_accuracy}, elapsed time:{datetime.now() - t0}')
                history['train_cost'].append(cost)
                history['train_accuracy'].append(accuracy)
                history['test_accuracy'].append(test_accuracy)
            if self.quadratic_logits:
                self._fitted_params = session.run(
                    [self.We, self.W2hl, self.W2hr, self.W2hlr, self.Whl, self.Whr, self.bh, self.Wo, self.bo])
            else:
                self._fitted_params = session.run([self.We, self.Whl, self.Whr, self.bh, self.Wo, self.bo])

        return history

    # def predict(self, words, left, right, lab):
    #     return self.session.run(
    #         self.prediction_op,
    #         feed_dict={
    #             self.words: words,
    #             self.left: left,
    #             self.right: right,
    #             self.labels: lab
    #         }
    #     )
    #
    # def score(self, ensamble):
    #     n_total = len(ensamble)
    #     n_correct = 0
    #     for words, left, right, lab in ensamble:
    #         p = self.predict(words, left, right, lab)
    #         n_correct += (p[-1] == lab[-1])
    #     return float(n_correct) / n_total
    #
    # def f1_score(self, ensamble):
    #     Y = []
    #     P = []
    #     for words, left, right, lab in ensamble:
    #         p = self.predict(words, left, right, lab)
    #         Y.append(lab[-1])
    #         P.append(p[-1])
    #     return f1_score(Y, P, average=None).mean()



    def preprocess_tree_data_for_model(self, tree, level=-1, path='*', label_modifier=None):
        def level_print(txt):
            print(path, txt)
        #level_print('')

        level = level + 1
        if tree is None:
            return [], [], [], []

        words_left, left_childs_left, right_childs_left, labels_left = self.preprocess_tree_data_for_model(tree.left, level=level, path=path + '<')
        words_right, left_childs_right, right_childs_right, labels_right = self.preprocess_tree_data_for_model(tree.right, level=level, path=path + '>')

        left_childs_count = len(words_left)
        right_childs_count = len(words_right)

        new_left_childs_right = [i if i == -1 else i + left_childs_count for i in left_childs_right]
        left = -1 if tree.left is None else left_childs_count - 1
        left_childs = left_childs_left + new_left_childs_right + [left]

        new_right_childs_right = [i if i == -1 else i + left_childs_count for i in right_childs_right]
        right = -1 if tree.right is None else left_childs_count + right_childs_count - 1
        right_childs = right_childs_left + new_right_childs_right + [right]

        tree_root_word = -1 if tree.word is None else tree.word
        words = words_left + words_right + [tree_root_word]

        labels = labels_left + labels_right + [tree.label]

        if label_modifier is not None:
            labels = [label_modifier(l) for l in labels]

        return words, left_childs, right_childs, labels
