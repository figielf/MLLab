import itertools

import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

from applications.CV.resnet_classifier import resnet_classifier
from tests.utils.cv_data_utils import get_blood_cell_images_info


def plot_history(history):
    print(history.history.keys())
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


def get_confusion_matrix(data_path, N, model, batch_size, image_size):
    # we need to see the data in the same order
    # for both predictions and targets
    print(f'Generating confusion matrix of {N} data from {data_path}')
    predictions = []
    targets = []
    i = 0
    data_generetor = model.scale_only_generator.flow_from_directory(
        data_path,
        target_size=image_size,
        shuffle=False,
        batch_size=batch_size * 2)
    labels = [c for c, id in sorted(data_generetor.class_indices.items(), key=lambda item: item[1])]
    for x, y in data_generetor:
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


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues,
                          show_plot=False):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    if show_plot:
        plt.show()


if __name__ == '__main__':
    EPOCHS = 16
    BATCH_SIZE = 128
    LEARNING_RATE = 0.0001

    image_size, train_path, test_path, train_images_files, test_image_files, classes = get_blood_cell_images_info(print_example=False)

    model = resnet_classifier(n_classes=len(classes), image_size=image_size, n_conv_blocks=3)
    history, checkpoint_filepath = model.fit(train_path, test_path, train_images_files, test_image_files, LEARNING_RATE, EPOCHS, BATCH_SIZE, plot_sample_preprocessed=False)

    plot_history(history)

    model.model.load_weights(checkpoint_filepath)

    cm, labels = get_confusion_matrix(train_path, len(train_images_files), model, BATCH_SIZE, image_size)
    test_cm, test_labels = get_confusion_matrix(test_path, len(test_image_files), model, BATCH_SIZE, image_size)

    # plot some data
    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plot_confusion_matrix(cm, labels, title='Train confusion matrix')
    plt.subplot(1, 2, 2)
    plot_confusion_matrix(test_cm, test_labels, title='Test confusion matrix')
    plt.show()



