import numpy as np
import tensorflow as tf
from sklearn.utils import shuffle

Normal = tf.compat.v1.distributions.Normal
Bernoulli = tf.compat.v1.distributions.Bernoulli

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


class variational_autoencoder_for_binary_variable:
    # this autoencoder works only for inputs with binary features
    # to extend it please change loss function accordingly so that it supports not binary inputs as targets

    class dense_layer(object):
        def __init__(self, input_dim, output_dim, f=tf.nn.relu):
            self.W = tf.Variable((np.random.randn(input_dim, output_dim) * 2 / np.sqrt(input_dim)).astype(np.float32))
            self.b = tf.Variable((np.zeros(output_dim)).astype(np.float32))
            self.activation = f

        def forward(self, X):
            return self.activation(tf.matmul(X, self.W) + self.b)

    def __init__(self, D, hidden_layer_sizes):
        self.D = D
        self.n_hiddens = len(hidden_layer_sizes)
        self.M = hidden_layer_sizes[-1]
        self._configure_layers(D, hidden_layer_sizes)
        self._session = None

        self._x_input = tf.compat.v1.placeholder(tf.float32, shape=(None, self.D))

        self.mu, sigma = self._encode(self._x_input)
        self.z_given_x = self._draw_z(tf.shape(self.mu)[0], self.mu, sigma)

        x_hat_logits = self._decode_to_logits(self.z_given_x)

        binary_cross_entropy = -tf.reduce_sum(
            input_tensor=tf.nn.sigmoid_cross_entropy_with_logits(
                labels=self._x_input,
                logits=x_hat_logits,
            ),
            axis=1
        )

        # Kullback-Leibler divergence between two Normal distributions
        # https://stats.stackexchange.com/questions/7440/kl-divergence-between-two-univariate-gaussians
        # https://stats.stackexchange.com/questions/60680/kl-divergence-between-two-multivariate-gaussians
        kl = -tf.compat.v1.log(sigma) + 0.5 * (sigma ** 2 + self.mu ** 2) - 0.5
        # as Normal variable dimensions are independent we can sum it
        kullback_leibler_distance = tf.reduce_sum(kl, axis=1)

        # ELBO cost faunction asumes here that input ant reconstructed input are of Bernoulli distributions (have binary features)
        self._binary_normal_elbo_cost_op = tf.reduce_mean(binary_cross_entropy - kullback_leibler_distance)

        z_standard = self._draw_z(1)
        x_generated_logits = self._decode_to_logits(z_standard)
        self.x_hat_sample = self._draw_x_hat(x_generated_logits)
        self.x_hat_sample_prob = tf.nn.sigmoid(x_generated_logits)

        self._z_input = tf.compat.v1.placeholder(tf.float32, shape=(None, self.M))
        prior_logits = self._decode_to_logits(self._z_input)
        self.prior_probs = tf.nn.sigmoid(prior_logits)

    def _encode(self, x, smoothing=1e-6):
        # calculate prior params distributions and extract z distribution params
        h = x
        for layer in self.encoder_layers:
            h = layer.forward(h)
        z_mu = h[:, :self.M]
        z_sigma = tf.nn.softplus(h[:, self.M:]) + smoothing
        return z_mu, z_sigma

    def _draw_z(self, n, mu=None, sigma=None):
        # sample from z prior distribution
        z_distib_standard = Normal(loc=np.zeros(self.M, dtype=np.float32), scale=np.ones(self.M, dtype=np.float32))
        sample = z_distib_standard.sample(n)
        if sigma is not None:
            sample = sample * sigma
        if mu is not None:
            sample = sample + mu
        return sample

    def _decode_to_logits(self, z):
        # calculate posterior sample
        x_hat_logits = z
        for layer in self.decoder_layers:
            x_hat_logits = layer.forward(x_hat_logits)
        return x_hat_logits

    def _draw_x_hat(self, logits):
        # sample from Bernoulli distribution
        x_hat_distib = Bernoulli(logits=logits)
        return x_hat_distib.sample()

    def _configure_layers(self, D, hidden_layer_sizes):
        self.encoder_layers = []
        self.decoder_layers = []

        # encoder hidden layers
        prev_out_size = D
        for layer_size in hidden_layer_sizes[:-1]:
            self.encoder_layers.append(self.dense_layer(prev_out_size, layer_size))
            prev_out_size = layer_size

        # final encoder layer with doubled param size (for mu and sigma), this uses identity activation
        h = self.dense_layer(prev_out_size, 2 * hidden_layer_sizes[-1], f=lambda x: x)
        self.encoder_layers.append(h)

        # decoder hidden layers
        prev_in_size = hidden_layer_sizes[-1]
        for layer_size in reversed(hidden_layer_sizes[:-1]):
            self.decoder_layers.append(self.dense_layer(prev_in_size, layer_size))
            prev_in_size = layer_size

        # final decoder layer with identity activation
        h = self.dense_layer(prev_in_size, self.D, f=lambda x: x)
        self.decoder_layers.append(h)

    def set_session(self, session):
        self._session = session

    def fit(self, X, learning_rate=0.001, n_epochs=30, batch_size=64):
        N, D = X.shape
        assert self.D == D
        n_batches = N // batch_size

        _train_op = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(-self._binary_normal_elbo_cost_op)
        history = []
        self.set_session(tf.compat.v1.Session())
        self._session.run(tf.compat.v1.global_variables_initializer())
        for i in range(n_epochs):
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_size: (j + 1) * batch_size]
                _, c = self._session.run((_train_op, self._binary_normal_elbo_cost_op), feed_dict={self._x_input: batch})
                if j % 100 == 0:
                    print(f'epoch:{i}, batch:{j}, cost: {c}')
                history.append(c)
        return history

    def transform(self, X):
        return self._session.run(self.mu, feed_dict={self._x_input: X})

    def prior_predictive_with_input(self, Z):
        return self._session.run(self.prior_probs, feed_dict={self._z_input: Z})

    def prior_predictive_sample_with_probs(self):
        # returns a sample from p(x_new | z), z ~ N(0, 1)
        return self._session.run((self.x_hat_sample, self.x_hat_sample_prob))

    def posterior_predictive_sample(self, X):
        # returns a sample from p(x_new | X)
        return self._session.run(self.x_hat_sample_prob, feed_dict={self._x_input: X})
