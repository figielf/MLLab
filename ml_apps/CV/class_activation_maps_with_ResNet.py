import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
from tensorflow.keras.models import Model


from CV.cv_data_utils import get_elephant_picture_as_array


def scale_img(x):
    x = x - x.min()
    x = x / x.max()
    return x


def build_activation_maps_utils_from_resnet(input_shape):
    resnet = ResNet50(input_shape=input_shape, weights='imagenet', include_top=True)
    resnet.trainable = False
    print('\nFull ResNet50 model summary:')
    resnet.summary()

    last_conv = resnet.layers[-3]  # output shape=(7, 7, 2048)
    classification_layer = resnet.layers[-1]  # output shape=(1000)
    classification_layer_weights = np.array(classification_layer.get_weights())
    weights = classification_layer_weights[0]  # only weights, biases are needless
    prediction_model = Model(resnet.input, [last_conv.output, resnet.output])  # output shape=(7, 7, 2048)
    return prediction_model, weights


if __name__ == '__main__':
    show = False
    input_shape = (224, 224, 3)

    model, w = build_activation_maps_utils_from_resnet(input_shape)

    image = get_elephant_picture_as_array(shape=input_shape, show=show)
    image_processed = preprocess_input(np.expand_dims(image, axis=0))
    fmaps, predictions = model.predict(image_processed)

    class_id = predictions[0].argmin()
    activation_map = fmaps[0].dot(w[:, class_id])
    activation_map = sp.ndimage.zoom(activation_map, (32, 32), order=1)  # reshape image from (7, 7) to (224, 224) so multiply by 32

    classnames = decode_predictions(predictions)
    print(classnames)
    classname = classnames[0][0][1]

    plt.figure(figsize=(20, 10))
    plt.subplot(1, 3, 1)
    plt.imshow(activation_map, cmap='jet', alpha=0.5)
    plt.colorbar()
    plt.subplot(1, 3, 2)
    plt.imshow(scale_img(image), alpha=0.8)
    plt.imshow(activation_map, cmap='jet', alpha=0.5)
    plt.subplot(1, 3, 3)
    plt.imshow(scale_img(image))
    plt.title(f'{classname} (prob: {predictions[0][class_id]})')
    plt.show()
