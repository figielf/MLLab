import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle
from sklearn.metrics import f1_score, accuracy_score
from sklearn.tree import DecisionTreeClassifier

from markov_models.hmm_discrete import hmm_discrete
from markov_models.mm_1order import SimpleMarkovModel
from tests.utils.nlp_data_utils import get_conll2000_data

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


class LogisticRegressionSparse:
    def __init__(self, D, K):
        self.D = D
        self.K = K
        self.W = None
        self.b = None

    def _init_params(self, init_params=None):
        if init_params is None:
            W_init = (np.random.randn(self.D, self.K) / np.sqrt(self.D + self.K)).astype(np.float32)
            b_init = np.zeros(self.K).astype(np.float32)
        else:
            W_init, b_init = init_params

        return tf.Variable(W_init), tf.Variable(b_init)

    def fit(self, X, Y, learning_rate=0.1, mu=0.99, batch_size=100, n_epochs=10, init_params=None):
        N = len(X)
        #self.D = len(set(X))
        #self.K = len(set(Y))
        n_batches = N // batch_size

        tf_W, tf_b = self._init_params(init_params)

        x_input = tf.compat.v1.placeholder(tf.int32, shape=(None))
        y_input = tf.compat.v1.placeholder(tf.int32, shape=(None))

        y_hat_logits = tf.nn.embedding_lookup(params=tf_W, ids=x_input) + tf_b
        y_hat = tf.argmax(input=y_hat_logits, axis=1)
        cost_op = tf.reduce_mean(
            input_tensor=tf.nn.sparse_softmax_cross_entropy_with_logits(
                labels=y_input,
                logits=y_hat_logits,
            )
        )

        train_op = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=mu).minimize(cost_op)
        # train_op = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost_op)

        history = []
        with tf.compat.v1.Session() as session:
            session.run(tf.compat.v1.global_variables_initializer())
            for i in range(n_epochs):
                X, Y = shuffle(X, Y)
                for j in range(n_batches):
                    x_batch = X[j * batch_size: (j * batch_size + batch_size)]
                    y_batch = Y[j * batch_size: (j * batch_size + batch_size)]
                    _, train_cost, train_pred = session.run((train_op, cost_op, y_hat),
                                                                  feed_dict={x_input: x_batch, y_input: y_batch})
                    train_error = self._error_rate(train_pred, y_batch)

                print(f'epoch:{i} - accuracy :{1 - train_error}, train cost:{train_cost}')
                history.append((train_cost, train_error))
            self.W, self.b = session.run((tf_W, tf_b))

        return history

    def _error_rate(self, p, t):
        return np.mean(p != t)

    def predict(self, X):
        logits = self.W[X] + self.b
        return np.argmax(logits, axis=1)


class HMM_POS:
    def __init__(self, V, K):
        self.V = V
        self.K = K
        self.pi = None
        self.A = None
        self.B = None

    def fit(self, X, Y, smoothing=10e-2):
        # sates matrix estimation
        mm = SimpleMarkovModel(self.K)
        mm.fit(Y, smoothing)
        self.pi = np.exp(mm.log_pi)
        self.A = np.exp(mm.log_A)

        # emission probabilities estimation
        b = np.ones((self.K, self.V)) * smoothing
        for x, y in zip(X, Y):
            for xi, yi in zip(x, y):
                b[yi, xi] += 1
        self.B = b / b.sum(axis=1, keepdims=True)

    def predict(self, X):
        hmm = hmm_discrete(self.K)
        hmm.pi = self.pi
        hmm.A = self.A
        hmm.B = self.B

        z = []
        for x in X:
            z.append(hmm.get_veterbi_hidden_states_sequence(x))
        return z

    def accuracy(self, p, t):
        t = np.concatenate(t)
        p = np.concatenate(p)
        return accuracy_score(t, p)

    def f1_score(self, p, t):
        t = np.concatenate(t)
        p = np.concatenate(p)
        return f1_score(t, p, average=None).mean()


def run_simple_model(model, Xtrain, Ytrain, Xtest, Ytest, model_label):
    model.fit(Xtrain, Ytrain)

    p = model.predict(Xtrain[:100])
    print('Train set size:', len(Xtrain))
    print(f'{model_label} train accuracy: {accuracy_score(Ytrain[:100], p)}')
    print(f'{model_label} train f1: {f1_score(Ytrain[:100], p, average=None).mean()}')

    p_test = model.predict(Xtest)
    print('Test set size:', len(Xtest))
    print(f'{model_label} test accuracy: {accuracy_score(Ytest, p_test)}')
    print(f'{model_label} test f1: {f1_score(Ytest, p_test, average=None).mean()}')


def run_seq_model(model, Xtrain, Ytrain, Xtest, Ytest, model_label):
    model.fit(Xtrain, Ytrain, smoothing=1e-1)

    p = model.predict(Xtrain)
    print('Train set size:', len(Xtrain))
    print(f'{model_label} train accuracy: {hmm.accuracy(Ytrain, p)}')
    print(f'{model_label} train f1: {hmm.f1_score(Ytrain, p)}')

    p_test = model.predict(Xtest)
    print('Test set size:', len(Xtest))
    print(f'{model_label} test accuracy: {hmm.accuracy(Ytest, p_test)}')
    print(f'{model_label} test f1: {hmm.f1_score(Ytest, p_test)}')


if __name__ == '__main__':
    # baseline
    Xtrain, Ytrain, Xtest, Ytest, word2idx = get_conll2000_data()
    Xtrain = np.array(Xtrain)
    Ytrain = np.array(Ytrain)
    Xtest = np.array(Xtest)
    Ytest = np.array(Ytest)
    N = len(Xtrain)
    V = len(word2idx) + 1
    K = len(set(Ytrain))
    print("vocabulary size:", V)
    print("num of POS tags:", K)

    dt = DecisionTreeClassifier()
    run_simple_model(dt, Xtrain.reshape(-1, 1).copy(), Ytrain.copy(), Xtest.reshape(-1, 1).copy(), Ytest.copy(), model_label='DecisionTreeClassifier')

    lr = LogisticRegressionSparse(V, K)
    run_simple_model(lr, Xtrain.copy(), Ytrain.copy(), Xtest.copy(), Ytest.copy(), model_label='LogisticRegressionSparse')

    # HMM with observed hidden POS states
    Xtrain_seq, Ytrain_seq, Xtest_seq, Ytest_seq, word2idx = get_conll2000_data(split_sequences=True)

    hmm = HMM_POS(V, K)
    run_seq_model(hmm, Xtrain_seq.copy(), Ytrain_seq.copy(), Xtest_seq.copy(), Ytest_seq.copy(), model_label='HMM')

