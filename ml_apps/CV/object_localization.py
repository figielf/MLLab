import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from skimage.transform import resize

from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.optimizers import Adam

from CV.cv_data_utils import get_pokemon_picture_as_array


def plot_image_with_red_box(image, boxes, height=100, width=100, show=True, ax=None):
    if show:
        fig, ax = plt.subplots(1)
    ax.imshow(image)
    for b in boxes:
        rect = Rectangle(
            (b[1] * width, b[0] * height),
            width=b[3] * width,
            height=b[2] * height,
            linewidth=1,
            edgecolor='r',
            facecolor='none')
        ax.add_patch(rect)
    if show:
        plt.show()


def data_generator(batch_size, height=100, width=100):
    while True:
        imgs, boxs = IMG_GENERATOR(batch_size, height=height, width=width)
        yield imgs, boxs


def build_simple_model(input_shape):
    vgg = vgg16.VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
    vgg.trainable = False

    x = Flatten()(vgg.output)
    out = Dense(4, activation='sigmoid')(x)

    model = Model(vgg.input, out)
    model.summary()
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001))
    return model


def check_predictions(model, n, height=100, width=100):
    batch_size = n**2
    imgs, target_boxs = IMG_GENERATOR(batch_size)
    predicted_boxes = model.predict(imgs)

    for t, p in zip(target_boxs, predicted_boxes):
        print('target coordinates:', t)
        print('predicted coordinates:', p)

    plt.figure(figsize=(20, 20))
    for i in range(batch_size):
        ax = plt.subplot(n, n, i + 1)
        plot_image_with_red_box(imgs[i], [predicted_boxes[i]], height=height, width=width, show=False, ax=ax)
        p_text = f'p:[{int(p[0] * height)}, {int(p[1] * width)}, {int(p[2] * height)}, {int(p[3] * width)}]'
        t_text = f't:[{int(t[0] * height)}, {int(t[1] * width)}, {int(t[2] * height)}, {int(t[3] * width)}]'
        ax.set_title(f'{p_text}\n{t_text}')
    plt.show()


def plot_img_example():
    img, box = IMG_GENERATOR(batch_size=1, height=IMG_HEIGHT, width=IMG_WIDTH)
    plot_image_with_red_box(img[0], [box[0]], height=IMG_HEIGHT, width=IMG_WIDTH)


def fit_model(model, n_epochs, steps_per_epoch):
    plot_img_example()
    model.fit(data_generator(BATCH_SIZE), epochs=n_epochs, steps_per_epoch=steps_per_epoch)
    check_predictions(model, n=3, height=IMG_HEIGHT, width=IMG_WIDTH)


def generate_image__white_rectangle_on_black_background(batch_size, height=100, width=100):
    images_batch = np.zeros((batch_size, height, width, 3))
    targets = np.zeros((batch_size, 4))
    for i in range(batch_size):
        h = np.random.randint(1, height)
        w = np.random.randint(1, width)
        x = np.random.randint(height - h)
        y = np.random.randint(width - w)
        images_batch[i, x:x+h, y:y+w, :] = 1

        targets[i, 0] = x / np.float32(height)
        targets[i, 1] = y / np.float32(width)
        targets[i, 2] = h / np.float32(height)
        targets[i, 3] = w / np.float32(width)
    return images_batch, targets


def generate_image__pokemon_on_black_background(batch_size, height=100, width=100):
    images_batch = np.zeros((batch_size, height, width, 3))
    targets = np.zeros((batch_size, 4))
    obj_h, obj_w, _ = POKEMON_IMG.shape
    for i in range(batch_size):
        obj = POKEMON_IMG[:, :, :3]

        if FLIP:
            if np.random.random() < 0.5:
                obj = np.fliplr(obj)

        if SCALE:
            scale = 0.5 + np.random.random()
            h = int(obj_h * scale)
            w = int(obj_w * scale)
            obj = resize(obj, (h, w), preserve_range=True).astype(np.uint8)
        else:
            h = obj_h
            w = obj_w

        x = np.random.randint(height - h)
        y = np.random.randint(width - w)
        images_batch[i, x:x+h, y:y+w, :] = obj

        targets[i, 0] = x / np.float32(height)
        targets[i, 1] = y / np.float32(width)
        targets[i, 2] = h / np.float32(height)
        targets[i, 3] = w / np.float32(width)
    return images_batch / 255., targets


if __name__ == '__main__':
    IMG_HEIGHT = 100
    IMG_WIDTH = 100
    BATCH_SIZE = 64
    FLIP = True
    SCALE = True

    POKEMON_IMG = get_pokemon_picture_as_array(show=True)
    print(POKEMON_IMG.shape)
    IMG_GENERATOR = generate_image__pokemon_on_black_background

    fit_model(build_simple_model(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)), n_epochs=5, steps_per_epoch=50)
