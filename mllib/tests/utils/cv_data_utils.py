import os
from glob import glob

import numpy as np
from tensorflow.keras.preprocessing import image
from matplotlib import pyplot as plt

from tests.consts import TEST_LARGE_DATA_PATH


def get_blood_cell_images_info(print_example=True):
    blood_cell_data_folder = os.path.join(TEST_LARGE_DATA_PATH, 'blood_cell_images')

    train_data_path = os.path.join(blood_cell_data_folder, 'TRAIN')
    test_data_path = os.path.join(blood_cell_data_folder, 'TEST')
    #simple_test_data_path = os.path.join(blood_cell_data_folder, 'TEST_SIMPLE')

    train_images_files = glob(train_data_path + '/*/*.jp*g')
    test_image_files = glob(test_data_path + '/*/*.jp*g')
    #simple_test_image_files = glob(simple_test_data_path + '/*/*.jp*g')
    folders = glob(train_data_path + '/*')

    if print_example:
        plt.imshow(image.load_img(np.random.choice(train_images_files)))
        plt.show()

    image_size = [224, 224]

    return image_size, train_data_path, test_data_path, train_images_files, test_image_files, folders