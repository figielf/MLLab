import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import fmin_l_bfgs_b
import tensorflow as tf
from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import AveragePooling2D, MaxPooling2D, Conv2D
from tensorflow.python.ops.numpy_ops import np_config

from CV.cv_data_utils import get_elephant_picture_as_array, get_style_picture_as_array

np_config.enable_numpy_behavior()
tf.compat.v1.enable_eager_execution()


def preprocess_image_for_VGG(image, show=False):
    image_processed = vgg16.preprocess_input(image)
    if show:
        plt.imshow(image_processed)
        plt.show()
    return image_processed


def build_base_vgg_model(input_shape):
    vgg = vgg16.VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
    vgg.trainable = False

    i = vgg.input
    x = i
    for layer in vgg.layers:
        if layer.__class__ == MaxPooling2D:
            # we want to account for features across the entire image
            # so get rid of the maxpool which throws away information
            x = AveragePooling2D()(x)
        else:
            x = layer(x)

    model = Model(i, x)
    return model


def build_content_model(input_shape, output_level=None, base_model=None):
    if not base_model:
        base_model = build_base_vgg_model(input_shape)

    conv_layers = []
    for layer in base_model.layers:
        if layer.__class__ == Conv2D:
            conv_layers.append(layer)

    if output_level is None:
        output_level = len(conv_layers) - 1

    assert 0 < output_level < len(conv_layers) + 1

    content_layer = conv_layers[output_level]
    print(f'Content model has potential {len(conv_layers)} Conv layers to represent image content and {output_level} was chosen which is {content_layer} layer.')

    model = Model(base_model.input, content_layer.get_output_at(0))
    return model


def build_style_model(input_shape, base_model=None):
    if not base_model:
        base_model = build_base_vgg_model(input_shape)

    conv_layer_outputs = []
    for layer in base_model.layers:
        if layer.name.endswith('conv1'):
            conv_layer_outputs.append(layer.get_output_at(1))

    model = Model(base_model.input, conv_layer_outputs)
    return model


def style_loss(y, t):
  return K.mean(K.square(gram_matrix(y) - gram_matrix(t)))


def vgg16_unpreprocess(img):
    img[..., 0] += 103.939
    img[..., 1] += 116.779
    img[..., 2] += 126.68
    img = img[..., ::-1]
    return img


def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x


def run_content_model(image, show, n_epochs=10):
    content_model = build_content_model(image.shape, output_level=12)
    #content_model.summary()

    elephant_image = np.expand_dims(image, axis=0)
    elephant_image = preprocess_image_for_VGG(elephant_image, show=show)
    target = tf.Variable(content_model.predict(elephant_image), trainable=False)

    @tf.function
    def get_content_loss_and_grads_func(x):
        y = content_model(x)
        loss = tf.math.reduce_mean(tf.math.square(target - y))
        grads = tf.gradients(loss, x)
        return [loss] + grads

    generated_content_image, losses = minimize(get_content_loss_and_grads_func, elephant_image.shape, n_epochs=n_epochs, maxfun=20, trace=True)
    plt.plot(losses)
    plt.show()

    final_generated_content_image = vgg16_unpreprocess(generated_content_image[0])
    plt.imshow(scale_img(final_generated_content_image))
    plt.show()


def run_style_model(image, show, n_epochs=10):
    style_model = build_style_model(image.shape)
    style_model.summary()

    style_image = np.expand_dims(image, axis=0)
    style_image = preprocess_image_for_VGG(style_image, show=show)
    style_layer_predictions = style_model.predict(style_image)

    targets = []
    for pred in style_layer_predictions:
        targets.append(calc_gram_matrix(tf.Variable(pred, trainable=False)))

    @tf.function
    def get_style_loss_and_grads_func(x):
        style_predictions = style_model(x)
        y = []
        for pred in style_predictions:
            y.append(calc_gram_matrix(pred))
        layer_losses = [tf.math.reduce_mean(tf.math.square(t - p)) for t, p in zip(targets, y)]
        loss = tf.math.reduce_sum(layer_losses)
        grads = tf.gradients(loss, x)
        return [loss] + grads

    generated_style_image, losses = minimize(get_style_loss_and_grads_func, style_image.shape, n_epochs=n_epochs, maxfun=20, trace=True)
    plt.plot(losses)
    plt.show()

    final_generated_style_image = vgg16_unpreprocess(generated_style_image[0])
    plt.imshow(scale_img(final_generated_style_image))
    plt.show()


def run_style_transfer(content_image, style_image, content_details, style_weights, show, n_epochs=10):
    print(content_image.shape)
    print(style_image.shape)
    assert content_image.shape == style_image.shape
    image_shape = content_image.shape

    base_model = build_base_vgg_model(image_shape)
    base_model.summary()
    content_model = build_content_model(image_shape, output_level=content_details, base_model=base_model)
    #content_model.summary()
    style_model = build_style_model(image_shape, base_model=base_model)
    #style_model.summary()

    content_image = np.expand_dims(content_image, axis=0)
    style_image = np.expand_dims(style_image, axis=0)
    content_image = preprocess_image_for_VGG(content_image, show=show)
    style_image = preprocess_image_for_VGG(style_image, show=show)

    content_target = tf.Variable(content_model.predict(content_image), trainable=False)
    style_layer_predictions = style_model.predict(style_image)

    style_targets = []
    for pred in style_layer_predictions:
        style_targets.append(calc_gram_matrix(tf.Variable(pred, trainable=False)))

    @tf.function
    def get_style_transfer_loss_and_grads_func(x):
        content_prediction = content_model(x)
        style_predictions = style_model(x)

        content_loss = tf.math.reduce_mean(tf.math.square(content_target - content_prediction))

        y = []
        for pred in style_predictions:
            y.append(calc_gram_matrix(pred))
        layer_losses = [tf.math.reduce_mean(tf.math.square(t - p)) for t, p in zip(style_targets, y)]
        style_loss = tf.math.reduce_sum([l * w for l, w in zip(layer_losses, style_weights)])

        loss = content_loss + style_loss
        grads = tf.gradients(loss, x)
        return [loss] + grads

    generated_style_image, losses = minimize(get_style_transfer_loss_and_grads_func, style_image.shape, n_epochs=n_epochs, maxfun=20, trace=True)
    plt.plot(losses)
    plt.show()

    final_generated_style_image = vgg16_unpreprocess(generated_style_image[0])

    plt.figure(figsize=(16, 16))
    plt.subplot(2, 2, 1)
    plt.imshow(scale_img(content_image[0]))
    plt.title('Content')
    plt.subplot(2, 2, 2)
    plt.imshow(scale_img(style_image[0]))
    plt.title('Style')
    plt.subplot(2, 2, 3)
    plt.imshow(scale_img(final_generated_style_image))
    plt.title('Content with style transferred')
    plt.show()


def calc_gram_matrix(image):
    image_reshaped = tf.transpose(image[0], perm=(2, 0, 1))  # (H, W, C) -> (C, H, W)
    x = tf.reshape(image_reshaped, [image_reshaped.shape[0], -1])  # reshape (C, H, W) -> (C, H * W)
    return tf.experimental.numpy.dot(x, tf.transpose(x)) / image.get_shape().num_elements()


def minimize(func, input_shape, n_epochs=10, maxfun=20, trace=True):
    def get_loss_and_grads_wrapper(x_vec):
        x_tensor = tf.Variable(x_vec.reshape(*input_shape).astype(np.float32), trainable=False)
        loss, grads = func(x_tensor)
        l = loss.numpy().astype(np.float64)
        g = grads.numpy().astype(np.float64).flatten()
        if trace:
            print('stop loss:', l)
        return l, g

    losses = []
    x = np.random.randn(np.prod(input_shape))
    for i in range(n_epochs):
        x, loss, _ = fmin_l_bfgs_b(
            func=get_loss_and_grads_wrapper,
            x0=x,
            #bounds=[[-127, 127]] * len(x.flatten()),
            maxfun=20)

        losses.append(loss)
        x = np.clip(x, -127, 127)
        if trace:
            print(f'iteration: {i}, loss: {loss}')

    return x.reshape(*input_shape), losses


if __name__ == '__main__':
    show = False
    style_name = 'starrynight'

    object_image = get_elephant_picture_as_array(show=show)
    style_image = get_style_picture_as_array(name=style_name, shape=object_image.shape, show=show)

    #run_content_model(image=object_image, show=show, n_epochs=10)
    #run_style_model(image=style_image, show=show, n_epochs=10)

    style_weights = np.array([0.2, 0.4, 0.3, 0.5, 0.2])
    #style_weights /= style_weights.sum()

    run_style_transfer(content_image=object_image, style_image=style_image, content_details=9, style_weights=style_weights, show=show, n_epochs=10)
