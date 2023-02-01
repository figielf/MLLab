import glob
import os

import numpy as np
import pandas as pd
from imageio import imread
from tensorflow.keras.utils import load_img, img_to_array
from matplotlib import pyplot as plt

from data_utils import get_data_dir, split_by_train_size, get_large_files_data_dir


def plot_examples(x, y, cmap='gray', labels=None):
    plt.figure(figsize=(15, 15))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(x[i], cmap=cmap)
        if labels is None:
            plt.xlabel(y[i])
        else:
            plt.xlabel(labels[y[i]])
    plt.show()


def _get_mnist_data_raw():
    print("Reading in and transforming data...")
    df = pd.read_csv(get_data_dir('mnist.csv'))
    data = df.values
    X = data[:, 1:].astype(np.float32)
    Y = data[:, 0].astype(int)
    assert X.shape[1] == 28 * 28
    picture_shape = (28, 28)
    return X, Y, picture_shape


def get_mnist_data(train_size=0.8, should_plot_examples=True):
    assert train_size >= 0
    X, Y, picture_shape = _get_mnist_data_raw()
    X = np.divide(X, 255.0)  # data is from 0..255

    X_train, X_test, Y_train, Y_test = split_by_train_size(X, Y, train_size=train_size)
    if should_plot_examples:
        plot_examples(X.reshape((-1, *picture_shape)), Y, cmap='gray', labels=None)
    return X_train, X_test, Y_train, Y_test, picture_shape


def _get_fruits_data_raw():
    print("Reading in and transforming data...")
    df = pd.read_csv(get_large_files_data_dir('mnist.csv'))
    data = df.values
    X = data[:, 1:].astype(np.float32)
    Y = data[:, 0].astype(int)
    assert X.shape[1] == 28 * 28
    picture_shape = (28, 28)
    return X, Y, picture_shape


def get_fruits_data(train_size=0.8):
    # https://www.kaggle.com/moltean/fruits

    assert train_size >= 0
    X, Y, picture_shape = _get_fruits_data_raw()
    X = np.divide(X, 255.0)  # data is from 0..255

    X_train, X_test, Y_train, Y_test = split_by_train_size(X, Y, train_size=train_size)
    return X_train, X_test, Y_train, Y_test, picture_shape


def get_elephant_picture_as_array(shape=None, show=False):
    # https://www.kaggle.com/datasets/imbikramsaha/caltech-101 -> elephant -> image_0002.jpg

    elephant_image_path = os.path.join('caltech101', 'elephant', 'image_0002.jpg')
    image_path = get_large_files_data_dir(elephant_image_path)
    image = load_img(image_path, target_size=shape)
    if show:
        plt.imshow(image)
        plt.show()
    return img_to_array(image)


def get_style_picture_as_array(name, shape=None, show=False):
    style_image_path = os.path.join('styles', f'{name}.jpg')
    image_path = get_large_files_data_dir(style_image_path)
    image = load_img(image_path, target_size=shape)
    if show:
        plt.imshow(image)
        plt.show()
    return img_to_array(image)


def get_pokemon_pictures_as_array(show=False):
    images = {}
    pokemon_names = ['charmander-tight', 'bulbasaur-tight', 'squirtle-tight']
    for name in pokemon_names:
        image_path = os.path.join('object_localization', f'{name}.png')
        image_path = get_data_dir(image_path)
        image = imread(image_path)
        images[name] = np.array(image)
    if show:
        n = len(images)
        for i, (name, img) in enumerate(images.items()):
            plt.subplot(1, n, i + 1)
            plt.imshow(img)
            plt.title(name)
        plt.show()
    return images


def get_object_localization_background_images_as_array(show=False):
    images = []
    image_dir = get_data_dir(os.path.join('object_localization', 'backgrounds'))
    background_img_paths = glob.glob(f'{image_dir}/*')
    for image_path in background_img_paths:
        image = imread(image_path)
        images.append(np.array(image))
    if show:
        plt.figure(figsize=(16, 16))
        n = int(np.sqrt(len(images))) + 1
        for i, img in enumerate(images):
            plt.subplot(n, n, i + 1)
            plt.imshow(img)
        plt.show()
    return images
