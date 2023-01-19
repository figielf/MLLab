import os
from glob import glob
from pathlib import Path

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, BatchNormalization, ZeroPadding2D, MaxPool2D, Activation, Add
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


class resnet_classifier:
    def __init__(self, n_classes, image_size, n_conv_blocks):
        self.n_classes = n_classes
        self.image_size = image_size
        self.n_conv_blocks = n_conv_blocks

        self.model = None
        self.augmentation_generator = None
        self.scale_only_generator = None
        self.input = Input(shape=image_size + [3])
        self._build_model()

    def _build_model(self):
        x = self.input
        x = ZeroPadding2D(padding=(3, 3))(x)
        x = Conv2D(64, (7, 7), strides=(2, 2), padding='valid', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        x = ZeroPadding2D(padding=(1, 1))(x)
        x = MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)

        x = self._conv_block(x, 3, [64, 64, 256], strides=(1, 1))
        x = self._identity_block(x, 3, [64, 64, 256])
        x = self._identity_block(x, 3, [64, 64, 256])

        x = self._conv_block(x, 3, [128, 128, 512])
        x = self._identity_block(x, 3, [128, 128, 512])
        x = self._identity_block(x, 3, [128, 128, 512])
        x = self._identity_block(x, 3, [128, 128, 512])

        x = Flatten()(x)
        #x = Dense(1024, activation='relu')(x)
        output = Dense(self.n_classes, activation='softmax')(x)

        self.model = Model(self.input, output)
        self.model.summary()

    def _identity_block(self, input_, kernel_size, filters):
        f1, f2, f3 = filters

        x = Conv2D(f1, kernel_size=(1, 1), kernel_initializer='he_normal')(input_)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(f2, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(f3, kernel_size=(1, 1), kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        x = Add()([x, input_])
        x = Activation('relu')(x)
        return x

    def _conv_block(self, input_, kernel_size, filters, strides=(2, 2)):
        f1, f2, f3 = filters

        x = Conv2D(f1, kernel_size=(1, 1), strides=strides, kernel_initializer='he_normal')(input_)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(f2, kernel_size=kernel_size, padding='same', kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        x = Conv2D(f3, kernel_size=(1, 1), kernel_initializer='he_normal')(x)
        x = BatchNormalization()(x)

        shortcut = Conv2D(f3, kernel_size=(1, 1), strides=strides, kernel_initializer='he_normal')(input_)
        shortcut = BatchNormalization()(shortcut)

        x = Add()([x, shortcut])
        x = Activation('relu')(x)
        return x

    def fit(self, train_data_path, test_data_path, train_images_files, test_image_files, learning_rate, n_epochs, batch_size, plot_sample_preprocessed=False):
        def preprocess_input(x):
            return x / 127.5 - 1.0

        image_files_count = len(train_images_files)
        test_image_files_count = len(test_image_files)

        self.augmentation_generator = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            preprocessing_function=preprocess_input)

        self.scale_only_generator = ImageDataGenerator(
            preprocessing_function=preprocess_input)

        if plot_sample_preprocessed:
            self._plot_sample_preprocessed_image(self.scale_only_generator, train_data_path)

        train_generator = self.augmentation_generator.flow_from_directory(
            train_data_path,
            target_size=self.image_size,
            shuffle=True,
            batch_size=batch_size,
            class_mode='sparse')
        test_generator = self.scale_only_generator.flow_from_directory(
            test_data_path,
            target_size=self.image_size,
            shuffle=True,
            batch_size=batch_size,
            class_mode='sparse')

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(learning_rate=learning_rate), metrics=['accuracy'])

        checkpoint_filepath = './temp/resnet_classifier_checkpoint'
        history = self.model.fit(
            train_generator,
            validation_data=test_generator,
            epochs=n_epochs,
            steps_per_epoch=1, #image_files_count // batch_size,
            validation_steps=1, #test_image_files_count // batch_size,)
            callbacks=[
                EarlyStopping(
                    monitor='loss',
                    patience=3,
                    restore_best_weights=True),
                ModelCheckpoint(
                    filepath=checkpoint_filepath,
                    save_weights_only=True,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True),
            ])

        return history, os.path.abspath(checkpoint_filepath)

    def _plot_sample_preprocessed_image(self, generator, data_path):
        testing_generator = generator.flow_from_directory(
            data_path,
            target_size=self.image_size,
            class_mode='sparse')
        labels = [c for c, id in sorted(testing_generator.class_indices.items(), key=lambda item: item[1])]
        # plot one sample after processing by preprocess_input]
        for x, y in testing_generator:
            print("min:", x[0].min(), "max:", x[0].max())
            plt.title(labels[np.argmax(y[0])])
            plt.imshow(x[0])
            plt.show()
            break

    def predict(self, X):
        return self.model.predict(X)
