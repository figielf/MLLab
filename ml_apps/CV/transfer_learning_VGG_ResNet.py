import os
from glob import glob
from pathlib import Path

import numpy as np
import tensorflow.keras.applications as keras_models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Embedding, Input, LSTM, \
    GlobalMaxPool1D, Bidirectional, Flatten
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from data_utils import get_large_files_data_dir, plot_confusion_matrix

FRUITS_360_FOLDER = os.path.abspath(get_large_files_data_dir('fruits-360_dataset'))
FRUITS_360_SMALL_FOLDER = os.path.abspath(get_large_files_data_dir('fruits-360-small'))

TRAIN_PATH = os.path.join(FRUITS_360_SMALL_FOLDER, 'Training')
TEST_PATH = os.path.join(FRUITS_360_SMALL_FOLDER, 'Test')


def extract_limited_fruits_data():
    def mkdir(p):
        if not os.path.exists(p):
            os.mkdir(p)

    def link(src, dst):
        if not os.path.exists(dst):
            os.symlink(src, dst, target_is_directory=True)

    def unlink(p):
        if os.path.exists(p):
            if os.path.islink(p):
                os.unlink(p)
                print(f'unlinked {p}')

    classes = [
        'Apple Golden 1',
        'Avocado',
        'Lemon',
        'Mango',
        'Kiwi',
        'Banana',
        'Strawberry',
        'Raspberry'
    ]

    train_path_from = os.path.join(FRUITS_360_FOLDER, 'Training')
    valid_path_from = os.path.join(FRUITS_360_FOLDER, 'Test')

    mkdir(FRUITS_360_SMALL_FOLDER)
    mkdir(TRAIN_PATH)
    mkdir(TEST_PATH)

    for d in Path(TRAIN_PATH).iterdir():
        unlink(d)
    for d in Path(TEST_PATH).iterdir():
        unlink(d)

    for c in classes:
        link(train_path_from + '/' + c, TRAIN_PATH + '/' + c)
        link(valid_path_from + '/' + c, TEST_PATH + '/' + c)


def plot_history(history):
    plt.figure(figsize=(10, 10))
    plt.subplot(1, 2, 1)
    plt.title('Loss')
    plt.plot(history.history['loss'], label='train loss')
    plt.plot(history.history['val_loss'], label='val loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title('Accuracy')
    plt.plot(history.history['accuracy'], label='train acc')
    plt.plot(history.history['val_accuracy'], label='val acc')
    plt.legend()
    plt.show()


def get_confusion_matrix(data_path, N, generator, model, batch_size):
    # we need to see the data in the same order
    # for both predictions and targets
    print(f'Generating confusion matrix of {N} data from {data_path}')
    predictions = []
    targets = []
    i = 0
    data_geneaetor = generator.flow_from_directory(data_path, target_size=IMAGE_SIZE, shuffle=False, batch_size=batch_size * 2)
    labels = [c for c, id in sorted(data_geneaetor.class_indices.items(), key=lambda item: item[1])]
    for x, y in data_geneaetor:
        i += 1
        if i % 100 == 0:
            print(i)
        p = model.predict(x)
        p = np.argmax(p, axis=1)
        y = np.argmax(y, axis=1)
        predictions = np.concatenate((predictions, p))
        targets = np.concatenate((targets, y))
        if len(targets) >= N:
            break

    cm = confusion_matrix(targets, predictions)
    return cm, labels


def plot_sample_preprocessed_image(generator):
    testing_generator = generator.flow_from_directory(TEST_PATH, target_size=IMAGE_SIZE)
    labels = [c for c, id in sorted(testing_generator.class_indices.items(), key=lambda item: item[1])]
    # plot one sample after processing by preprocess_input]
    for x, y in testing_generator:
        print("min:", x[0].min(), "max:", x[0].max())
        plt.title(labels[np.argmax(y[0])])
        plt.imshow(x[0])
        plt.show()
        break


def run_model(base_model, preprocess_input_for_base_model, data_folder, n_epochs, fit_batch_size):

    image_files_count = len(glob(os.path.join(data_folder, 'Training') + '/*/*.jp*g'))
    test_image_files_count = len(glob(os.path.join(data_folder, 'Test') + '/*/*.jp*g'))
    folders_count = len(glob(os.path.join(data_folder, 'Training') + '/*'))

    n_classes = folders_count

    i = base_model.input
    x = Flatten()(base_model.output)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(i, x)
    model.summary()

    model.compile(
        loss='categorical_crossentropy',
        optimizer='rmsprop',
        metrics=['accuracy']
    )

    generator = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input_for_base_model)

    #plot_sample_preprocessed_image(generator)

    train_generator = generator.flow_from_directory(
        TRAIN_PATH,
        target_size=IMAGE_SIZE,
        shuffle=True,
        batch_size=fit_batch_size)
    test_generator = generator.flow_from_directory(
        TEST_PATH,
        target_size=IMAGE_SIZE,
        shuffle=True,
        batch_size=fit_batch_size)

    history = model.fit(
        train_generator,
        validation_data=test_generator,
        epochs=n_epochs,
        steps_per_epoch=image_files_count // fit_batch_size,
        validation_steps=test_image_files_count // fit_batch_size, )

    #plot_history(history)

    cm, labels = get_confusion_matrix(TRAIN_PATH, image_files_count, generator, model, fit_batch_size)
    print('Train data', cm)
    test_cm, test_labels = get_confusion_matrix(TEST_PATH, test_image_files_count, generator, model, fit_batch_size)
    print('Test data', cm)
    print(test_cm)

    # plot some data
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(cm, labels, title='Train confusion matrix')
    plt.subplot(1, 2, 2)
    plot_confusion_matrix(test_cm, test_labels, title='Test confusion matrix')
    #plt.show()


if __name__ == '__main__':
    extract_limited_fruits_data()

    IMAGE_SIZE = [100, 100]

    epochs = 5
    batch_size = 32

    print('\n\nUse VGG16 as base madel for transfer learning')
    vgg = keras_models.vgg16.VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    vgg.trainable = False
    vgg_preprocess = keras_models.vgg16.preprocess_input
    run_model(vgg, vgg_preprocess, FRUITS_360_SMALL_FOLDER, epochs, batch_size)


    print('\n\nUse ResNet50 as base madel for transfer learning')
    resnet = keras_models.resnet.ResNet50(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)
    resnet.trainable = False
    resnet_preprocess = keras_models.resnet.preprocess_input
    run_model(resnet, resnet_preprocess, FRUITS_360_SMALL_FOLDER, epochs, batch_size)

    plt.show()
