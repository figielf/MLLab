import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.utils import shuffle

if tf.__version__.startswith('2'):
    tf.compat.v1.disable_eager_execution()


def leaky_relu(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


class dcgan_tf1:
    class dense_layer:
        def __init__(self, name, input_dim, output_dim, use_batch_norm, f=tf.nn.relu):
            self.name = name
            self.W = tf.compat.v1.get_variable(
                f'W_{name}',
                shape=(input_dim, output_dim),
                initializer=tf.random_normal_initializer(stddev=0.02))
            self.b = tf.compat.v1.get_variable(
                f'b_{name}',
                shape=(output_dim,),
                initializer=tf.zeros_initializer())
            self.use_batch_norm = use_batch_norm
            self.activation = f
            #print(f'created name={self.name}, W.shape={(input_dim, output_dim)}, b.shape={(output_dim, )}')

        def forward(self, X, reuse, is_training):
            a = tf.matmul(X, self.W) + self.b

            if self.use_batch_norm:
                a = tf.compat.v1.layers.batch_normalization(
                    a,
                    momentum=0.9,
                    epsilon=1e-5,
                    scale=True,
                    training=is_training,
                    reuse=reuse,
                    name=self.name)
            return self.activation(a)

    class conv_layer:
        def __init__(self, name, input_feature_map_dim, output_feature_map_dim, use_batch_norm, filter_size=5, stride=2, f=tf.nn.relu):
            self.name = name
            self.W = tf.compat.v1.get_variable(
                f'W_{name}',
                shape=(filter_size, filter_size, input_feature_map_dim, output_feature_map_dim),
                initializer=tf.compat.v1.truncated_normal_initializer(stddev=0.02))
            self.b = tf.compat.v1.get_variable(
                f'b_{name}',
                shape=(output_feature_map_dim,),
                initializer=tf.zeros_initializer())
            self.use_batch_norm = use_batch_norm
            self.filter_size = filter_size
            self.stride = stride
            self.activation = f
            #print(f'created name={self.name}, W.shape={(filter_size, filter_size, input_feature_map_dim, output_feature_map_dim)}, b.shape={(output_feature_map_dim,)}')

        def forward(self, X, reuse, is_training):
            conv_out = tf.nn.conv2d(
              X,
              self.W,
              strides=[1, self.stride, self.stride, 1],
              padding='SAME')
            conv_out = tf.nn.bias_add(conv_out, self.b)

            if self.use_batch_norm:
                conv_out = tf.compat.v1.layers.batch_normalization(
                    conv_out,
                    momentum=0.9,
                    epsilon=1e-5,
                    scale=True,
                    training=is_training,
                    reuse=reuse,
                    name=self.name)
            return self.activation(conv_out)

    class fractionally_strided_conv_layer:
        def __init__(self, name, input_feature_map_dim, output_feature_map_dim, output_shape, use_batch_norm, filter_size=5, stride=2, f=tf.nn.relu):
            self.name = name
            self.W = tf.compat.v1.get_variable(
                f'W_{name}',
                shape=(filter_size, filter_size, output_feature_map_dim, input_feature_map_dim),
                initializer=tf.random_normal_initializer(stddev=0.02))
            self.b = tf.compat.v1.get_variable(
                f'b_{name}',
                shape=(output_feature_map_dim,),
                initializer=tf.zeros_initializer())
            self.output_shape = output_shape
            self.use_batch_norm = use_batch_norm
            self.filter_size = filter_size
            self.stride = stride
            self.activation = f
            #print(f'created name={self.name}, W.shape={(filter_size, filter_size, output_feature_map_dim, input_feature_map_dim)}, b.shape={(output_feature_map_dim,)}')

        def forward(self, X, reuse, is_training):
            conv_out = tf.compat.v1.nn.conv2d_transpose(
                value=X,
                filter=self.W,
                output_shape=self.output_shape,
                strides=[1, self.stride, self.stride, 1])
            conv_out = tf.nn.bias_add(conv_out, self.b)

            # apply batch normalization
            if self.use_batch_norm:
                conv_out = tf.compat.v1.layers.batch_normalization(
                    conv_out,
                    momentum=0.9,
                    epsilon=1e-5,
                    scale=True,
                    training=is_training,
                    reuse=reuse,
                    name=self.name)

            return self.activation(conv_out)

    def __init__(self, img_length, num_colors, discriminator_config, generator_config):
        self.img_length = img_length
        self.n_colors = num_colors
        self.discriminator_config = discriminator_config
        self.generator_config = generator_config
        self.latent_dims = generator_config['z']
        self._session = None

        self.batch_size = tf.compat.v1.placeholder(tf.int32, shape=(), name='batch_size')

        self._x_input = tf.compat.v1.placeholder(tf.float32, shape=(None, self.img_length, self.img_length, self.n_colors), name='X')
        self.Z = tf.compat.v1.placeholder(tf.float32, shape=(None, self.latent_dims), name='Z')

        self._configure_discriminator(self.discriminator_config)
        with tf.compat.v1.variable_scope("discriminator") as scope:
            discriminator_x_logits = self._discriminator_forward_logits(self._x_input, reuse=None, is_training=True)
        self._configure_generator(self.generator_config)
        with tf.compat.v1.variable_scope("generator") as scope:
            self.generated_images = self._generator_forward(self.Z, reuse=None, is_training=True)

        with tf.compat.v1.variable_scope("discriminator") as scope:
            scope.reuse_variables()
            generated_images_logits = self._discriminator_forward_logits(self.generated_images, reuse=True, is_training=True)

        with tf.compat.v1.variable_scope("generator") as scope:
            scope.reuse_variables()
            self.generated_images_test = self._generator_forward(self.Z, reuse=True, is_training=False)


        # build costs
        discriminator_cost_real = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=discriminator_x_logits,
            labels=tf.ones_like(discriminator_x_logits)
        )
        discriminator_cost_fake = tf.nn.sigmoid_cross_entropy_with_logits(
            logits=generated_images_logits,
            labels=tf.zeros_like(generated_images_logits)
        )
        self._discriminator_cost = tf.reduce_mean(discriminator_cost_real) + tf.reduce_mean(discriminator_cost_fake)
        self._generator_cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=generated_images_logits,
                labels=tf.ones_like(generated_images_logits)
            )
        )

        real_predictions = tf.cast(discriminator_x_logits > 0, tf.float32)
        fake_predictions = tf.cast(generated_images_logits < 0, tf.float32)
        self._correct_predictions_count = tf.reduce_sum(real_predictions) + tf.reduce_sum(fake_predictions)


    def _configure_discriminator(self, layers_config):
        with tf.compat.v1.variable_scope("discriminator") as scope:

            # conv layers
            self.discriminator_conv_layers = []
            in_features = self.n_colors
            img_out_length = self.img_length
            for i, (out_features, filter_size, stride, apply_batch_norm) in enumerate(layers_config['conv_layers']):
                self.discriminator_conv_layers.append(
                    self.conv_layer(f'conv_layer_{i}', in_features, out_features, apply_batch_norm, filter_size, stride, leaky_relu))
                in_features = out_features
                img_out_length = int(np.ceil(float(img_out_length) / stride))


            # dense layers
            dense_in_size = in_features * img_out_length * img_out_length
            self.discriminator_dense_layers = []
            for i, (out_size, apply_batch_norm) in enumerate(layers_config['dense_layers']):
                self.discriminator_dense_layers.append(
                    self.dense_layer(f'dense_layer_{i}', dense_in_size, out_size, apply_batch_norm, leaky_relu))
                dense_in_size = out_size

            # last logistic layer
            name = f'logistic_dense_layer_{len(self.discriminator_dense_layers)}'
            self.discriminator_logistic_layer = self.dense_layer(name, dense_in_size, 1, False, lambda x: x)

    def _discriminator_forward_logits(self, x, reuse=None, is_training=True):
        y_hat = x
        for layer in self.discriminator_conv_layers:
            y_hat = layer.forward(y_hat, reuse, is_training)
        y_hat = tf.compat.v1.layers.flatten(y_hat)
        for layer in self.discriminator_dense_layers:
            y_hat = layer.forward(y_hat, reuse, is_training)
        y_hat = self.discriminator_logistic_layer.forward(y_hat, reuse, is_training)
        return y_hat

    def _configure_generator(self, layers_config):
        with tf.compat.v1.variable_scope("generator") as scope:

            conv_out_img_sizes = [self.img_length]
            conv_out_img_size = self.img_length
            for _, _, stride, _ in reversed(layers_config['conv_layers']):
                conv_out_img_size = int(np.ceil(float(conv_out_img_size) / stride))
                conv_out_img_sizes.append(conv_out_img_size)
            conv_in_img_sizes = list(reversed(conv_out_img_sizes))
            self._conv_input_features_count = conv_in_img_sizes[0]

            # dense layers
            dense_in_size = self.latent_dims
            self._generator_dense_layers = []
            for i, (out_size, apply_batch_norm) in enumerate(layers_config['dense_layers']):
                self._generator_dense_layers.append(
                    self.dense_layer(f'gen_dense_layer_{i}', dense_in_size, out_size, apply_batch_norm, tf.nn.relu))
                dense_in_size = out_size

            # final dense layer
            self._conv_in_features_count = layers_config['projection']
            dense_out_size = self._conv_in_features_count * self._conv_input_features_count * self._conv_input_features_count
            use_batch_norm_after_project = layers_config['bn_after_project']
            self._generator_dense_layers.append(self.dense_layer(
                f'gen_dense_layer_{len(self._generator_dense_layers)}',
                dense_in_size,
                dense_out_size,
                not use_batch_norm_after_project,
                tf.nn.relu))

            # conv layers
            in_features = self._conv_in_features_count

            activation_functions = [tf.nn.relu] * len(layers_config['conv_layers'])
            activation_functions[-1] = layers_config['output_activation']

            self._generator_conv_layers = []
            for i, (out_features, filter_size, stride, apply_batch_norm) in enumerate(layers_config['conv_layers']):
                out_shape = [self.batch_size, conv_in_img_sizes[i + 1], conv_in_img_sizes[i + 1], out_features]
                self._generator_conv_layers.append(self.fractionally_strided_conv_layer(
                    f'gen_fs_conv_layer_{i}',
                    in_features,
                    out_features,
                    out_shape,
                    apply_batch_norm,
                    filter_size,
                    stride,
                    activation_functions[i]))
                in_features = out_features

            self._use_batch_norm_after_project = use_batch_norm_after_project

    def _generator_forward(self, z, reuse=None, is_training=True):
        conv_input_features_count = self._conv_input_features_count
        use_batch_norm = self._use_batch_norm_after_project
        conv_in_features_count = self._conv_in_features_count
        gen_output = z
        for layer in self._generator_dense_layers:
            gen_output = layer.forward(gen_output, reuse, is_training)

        gen_output = tf.compat.v1.reshape(gen_output, [-1, conv_input_features_count, conv_input_features_count, conv_in_features_count])

        # apply batch norm
        if use_batch_norm:
            gen_output = tf.compat.v1.layers.batch_normalization(
                gen_output,
                decay=0.9,
                epsilon=1e-5,
                scale=True,
                training=is_training,
                reuse=reuse,
                name='bn_after_project'
            )

        for layer in self._generator_conv_layers:
            gen_output = layer.forward(gen_output, reuse, is_training)
        return gen_output

    def set_session(self, session):
        self._session = session

    def fit(self, X, learning_rate=0.0002, beta=0.5, n_epochs=2, batch_size=64, generate_samples_period=50):
        N = len(X)
        n_batches = N // batch_size

        num_predictions = 2.0 * batch_size
        discriminator_accuracy_op = self._correct_predictions_count / num_predictions

        discriminator_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith('discriminator')]
        generator_params = [t for t in tf.compat.v1.trainable_variables() if t.name.startswith('generator')]

        _discriminator_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=beta).minimize(
            self._discriminator_cost, var_list=discriminator_params)
        _generator_train_op = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1=beta).minimize(
            self._generator_cost, var_list=generator_params)

        history = {
            'discriminator_cost': [],
            'generator_cost': [],
            'generated_samples': []}
        self.set_session(tf.compat.v1.Session())  # tf.InteractiveSession()
        self._session.run(tf.compat.v1.global_variables_initializer())
        total_iterations = 0
        for i in range(n_epochs):
            X = shuffle(X)
            for j in range(n_batches):
                batch = X[j * batch_size: (j + 1) * batch_size]

                Z = np.random.uniform(-1, 1, size=(batch_size, self.latent_dims))

                _, discriminator_c, discriminator_accuracy = self._session.run(
                    (_discriminator_train_op, self._discriminator_cost, discriminator_accuracy_op),
                    feed_dict={self._x_input: batch, self.Z: Z, self.batch_size: batch_size})

                _, generator_c1 = self._session.run(
                    (_generator_train_op, self._generator_cost),
                    feed_dict={self.Z: Z, self.batch_size: batch_size})
                _, generator_c2 = self._session.run(
                    (_generator_train_op, self._generator_cost),
                    feed_dict={self.Z: Z, self.batch_size: batch_size})
                generator_c = (generator_c1 + generator_c2) / 2


                if j % 1 == 0:
                    print(f'epoch:{i}, batch:{j}/{n_batches}, discriminator accuracy:{discriminator_accuracy}, generator cost: {generator_c}, discriminator cost: {discriminator_c}')
                history['discriminator_cost'].append(discriminator_c)
                history['generator_cost'].append(generator_c)

                #generated_sample = self.sample(batch_size)
                #history['generated_samples'].append(generated_sample)

                total_iterations += 1
                if generate_samples_period is not None and total_iterations % generate_samples_period == 0:
                    self.save_generated_samples(total_iterations, folder_suffix='mnist')

        return history

    def sample(self, n):
        Z = np.random.uniform(-1, 1, size=(n, self.latent_dims))
        samples = self._session.run(self.generated_images_test, feed_dict={self.Z: Z, self.batch_size: n})
        return samples

    def save_generated_samples(self, total_iters, folder_suffix=None):
        # save samples periodically
        print(f'saving a sample at iteration {total_iters} ...')
        samples = self.sample(64)  # shape is (64, D, D, color)

        # for convenience
        d = self.img_length

        if samples.shape[-1] == 1:
            # if color == 1, we want a 2-D image (N x N)
            samples = samples.reshape(64, d, d)
            flat_image = np.empty((8 * d, 8 * d))

            k = 0
            for i in range(8):
                for j in range(8):
                    flat_image[i * d:(i + 1) * d, j * d:(j + 1) * d] = samples[k].reshape(d, d)
                    k += 1

            plt.imshow(flat_image, cmap='gray')
        else:
            # if color == 3, we want a 3-D image (N x N x 3)
            flat_image = np.empty((8 * d, 8 * d, 3))
            k = 0
            for i in range(8):
                for j in range(8):
                    flat_image[i * d:(i + 1) * d, j * d:(j + 1) * d] = samples[k]
                    k += 1
            plt.imshow(flat_image)

        folder_name = 'gan_samples'
        if folder_suffix is not None:
            folder_name += f'_{folder_suffix}'
        if not os.path.exists(folder_name):
            os.mkdir(folder_name)
        plt.savefig(f'{folder_name}/samples_at_iter_{total_iters}.png')
