import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from skimage.transform import resize

from tensorflow.keras.applications import vgg16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Flatten, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import binary_crossentropy, categorical_crossentropy

from CV.cv_data_utils import get_pokemon_pictures_as_array, get_object_localization_background_images_as_array


def plot_image_with_red_box(image, boxes, show=True, ax=None):
    height = IMG_HEIGHT
    width = IMG_WIDTH
    if show:
        fig, ax = plt.subplots(1)
    ax.imshow(image)
    for b in boxes:
        if b[-1] > 0.5:
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


def data_generator(batch_size):
    while True:
        imgs, boxs = IMG_GENERATOR(batch_size)
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


def build_model_with_appearance(input_shape):
    def custom_loss_with_appearance(y_true, y_pred):
        beta_coordinates = 2.0
        beta_appear = 0.5
        coordinates_loss = binary_crossentropy(y_true[:, :-1], y_pred[:, :-1])
        appear_loss = binary_crossentropy(y_true[:, -1], y_pred[:, -1])
        return beta_coordinates * coordinates_loss * y_true[:, -1] + beta_appear * appear_loss

    vgg = vgg16.VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
    vgg.trainable = False

    x = Flatten()(vgg.output)
    out = Dense(5, activation='sigmoid')(x)

    model = Model(vgg.input, out)
    model.summary()
    model.compile(loss=custom_loss_with_appearance, optimizer=Adam(learning_rate=0.001))
    return model


def build_model_with_appearance_and_object_classification(input_shape):
    def custom_loss_with_appearance_and_object_classification(y_true, y_pred):
        beta_coordinates = 1.0
        beta_classifier = 1.0
        beta_appear = 0.5
        coordinates_loss = binary_crossentropy(y_true[:, :4], y_pred[:, :4])
        classifier_loss = categorical_crossentropy(y_true[:, 4:-1], y_pred[:, 4:-1])
        appear_loss = binary_crossentropy(y_true[:, -1], y_pred[:, -1])
        return beta_coordinates * coordinates_loss * y_true[:, -1] \
               + beta_classifier * classifier_loss * y_true[:, -1] \
               + beta_appear * appear_loss

    vgg = vgg16.VGG16(input_shape=input_shape, weights='imagenet', include_top=False)
    vgg.trainable = False

    x = Flatten()(vgg.output)
    coordinates = Dense(4, activation='sigmoid')(x)
    object_class = Dense(3, activation='softmax')(x)
    appear = Dense(1, activation='sigmoid')(x)
    out = Concatenate()([coordinates, object_class, appear])

    model = Model(vgg.input, out)
    model.summary()
    model.compile(loss=custom_loss_with_appearance_and_object_classification, optimizer=Adam(learning_rate=0.001))
    return model


def check_predictions(model, n):
    height = IMG_HEIGHT
    width = IMG_WIDTH
    batch_size = n ** 2
    imgs, target = IMG_GENERATOR(batch_size)
    prediction = model.predict(imgs)

    for t, p in zip(target, prediction):
        print('target coordinates:', t)
        print('predicted coordinates:', p)

    plt.figure(figsize=(20, 20))
    for i in range(batch_size):
        ax = plt.subplot(n, n, i + 1)
        plot_image_with_red_box(imgs[i], [prediction[i]], show=False, ax=ax)

        if prediction[i, -1] > 0.5:
            predicted_class = np.argmax(prediction[i, 4:-1])
            p_pok_name = POKEMONS[predicted_class].replace('-tight', '')
            p_text = f'p:[{int(p[0] * height)}, {int(p[1] * width)}, {int(p[2] * height)}, {int(p[3] * width)}, {p_pok_name}]'
        else:
            p_text = 'p:[[no object]'

        if target[i, -1] > 0.5:
            target_class = np.argmax(target[i, 4:-1])
            t_pok_name = POKEMONS[target_class].replace('-tight', '')
            t_text = f't:[{int(t[0] * height)}, {int(t[1] * width)}, {int(t[2] * height)}, {int(t[3] * width)}, {t_pok_name}]'
        else:
            t_text = 't:[no object]'

        ax.set_title(f'{p_text}\n{t_text}')
    plt.show()


def plot_img_example():
    img, box = IMG_GENERATOR(batch_size=1)
    plot_image_with_red_box(img[0], [box[0]])


def fit_model(model, n_epochs, steps_per_epoch, show=True):
    if show:
        plot_img_example()
    model.fit(data_generator(BATCH_SIZE), epochs=n_epochs, steps_per_epoch=steps_per_epoch)
    check_predictions(model, n=3)


def select_pokemon():
    pokemon_id = np.random.randint(len(POKEMONS))
    pokemon_name = POKEMONS[pokemon_id]
    return POKEMON_IMGS[pokemon_name], pokemon_id


def select_background(height, width):
    full_background = BACKGROUNDS[np.random.randint(len(BACKGROUNDS))]
    h, w, _ = full_background.shape
    start_h = np.random.randint(h - height)
    end_h = start_h + height
    start_w = np.random.randint(w - width)
    return full_background[start_h:end_h, start_w:start_w + width].copy()


def calc_targets(x, y, h, w, height, width):
    t = np.zeros((4,))
    t[0] = x / np.float32(height)
    t[1] = y / np.float32(width)
    t[2] = h / np.float32(height)
    t[3] = w / np.float32(width)
    return t


def generate_image__white_rectangle_on_black_background(batch_size):
    height = IMG_HEIGHT
    width = IMG_WIDTH
    images_batch = np.zeros((batch_size, height, width, 3))
    targets = np.zeros((batch_size, 5))
    for i in range(batch_size):
        appear = np.random.random() < P_APPEAR  # will pokemon appear on the image?
        if appear:
            h = np.random.randint(1, height)
            w = np.random.randint(1, width)
            x = np.random.randint(height - h)
            y = np.random.randint(width - w)

            images_batch[i, x:x + h, y:y + w, :] = 1

            targets[i, :4] = calc_targets(x, y, h, w, height, width)
        targets[i, 4] = appear
    return images_batch, targets


def generate_image__pokemon_on_black_background(batch_size):
    height = IMG_HEIGHT
    width = IMG_WIDTH
    images_batch = np.zeros((batch_size, height, width, 3))
    targets = np.zeros((batch_size, 8))
    for i in range(batch_size):
        appear = np.random.random() < P_APPEAR  # will pokemon appear on the image?
        if appear:
            pokemon_img, pokemon_id = select_pokemon()
            targets[i, 4 + pokemon_id] = 1.0

            obj_h, obj_w, _ = pokemon_img.shape
            obj = pokemon_img.copy()

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
            images_batch[i, x:x + h, y:y + w, :] = obj[:, :, :3]

            targets[i, :4] = calc_targets(x, y, h, w, height, width)
        targets[i, -1] = appear
    return images_batch / 255., targets


def generate_image__pokemon_with_background(batch_size):
    height = IMG_HEIGHT
    width = IMG_WIDTH
    images_batch = np.zeros((batch_size, height, width, 3))
    targets = np.zeros((batch_size, 8))
    for i in range(batch_size):
        background_img = select_background(height, width)
        assert background_img.shape == (height, width, 3)

        appear = np.random.random() < P_APPEAR  # will pokemon appear on the image?
        if appear:
            pokemon_img, pokemon_id = select_pokemon()
            targets[i, 4 + pokemon_id] = 1.0

            obj_h, obj_w, _ = pokemon_img.shape
            obj = pokemon_img.copy()

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

            object_mask = obj[:, :, 3] == 0
            object_mask = np.expand_dims(object_mask, -1)
            background_img[x:x + h, y:y + w, :] = background_img[x:x + h, y:y + w, :] * object_mask
            background_img[x:x + h, y:y + w, :] = background_img[x:x + h, y:y + w, :] + obj[:, :, :3]
            targets[i, :4] = calc_targets(x, y, h, w, height, width)

        images_batch[i] = background_img
        targets[i, -1] = appear
    return images_batch / 255., targets


if __name__ == '__main__':
    SHOW = False
    IMG_HEIGHT = 200
    IMG_WIDTH = 200
    BATCH_SIZE = 64
    FLIP = True
    SCALE = True
    P_APPEAR = 0.75
    IMG_GENERATOR = generate_image__pokemon_with_background
    POKEMONS = ['charmander-tight', 'bulbasaur-tight', 'squirtle-tight']
    N_POKEMONS = len(POKEMONS)

    POKEMON_IMGS = get_pokemon_pictures_as_array(show=SHOW)
    BACKGROUNDS = get_object_localization_background_images_as_array(show=SHOW)

    fit_model(
        build_model_with_appearance_and_object_classification(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        n_epochs=5,
        steps_per_epoch=50,
        show=SHOW)
