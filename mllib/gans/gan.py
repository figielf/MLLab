import os
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from tensorflow.keras.layers import Input, Dense, LeakyReLU, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD, Adam


class gan:
    def __init__(self, latent_dim, D):
        self.latent_dim = latent_dim
        self.D = D
        self.generator_model = None
        self.discriminator_model = None

        save_folder = 'gan_images'
        if not os.path.exists(save_folder):
            os.mkdir(save_folder)
        self._save_path = os.path.join(save_folder, datetime.now().strftime('%d%m%Y_%H%M%S'))

    def fit(self, X, n_epochs, batch_size, save_images_details=None, log_period=10):
        self.discriminator_model = self._build_discriminator_model()
        self.discriminator_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5), metrics=['accuracy'])

        self.generator_model = self._build_generator_model()

        self.discriminator_model.trainable = False
        z = Input(shape=(self.latent_dim, ))
        genarate_image = self.generator_model(z)
        fake_prediction = self.discriminator_model(genarate_image)
        full_pass_model = Model(z, fake_prediction)
        full_pass_model.compile(loss='binary_crossentropy', optimizer=Adam(0.0002, 0.5))

        ones = np.ones(batch_size)
        zeros = np.zeros(batch_size)

        history = {
            'discriminator_loss': [],
            'discriminator_accuracy': [],
            'generator_loss': []
        }

        for epoch in range(n_epochs):
            real_images = X[np.random.randint(0, X.shape[0], batch_size)]
            noise = np.random.randn(batch_size, self.latent_dim)
            fake_images = self.generator_model.predict(noise)

            loss_real, acc_real = self.discriminator_model.train_on_batch(real_images, ones)
            loss_fake, acc_fake = self.discriminator_model.train_on_batch(fake_images, zeros)
            history['discriminator_loss'].append(0.5 * (loss_real + loss_fake))
            history['discriminator_accuracy'].append(0.5 * (acc_real + acc_fake))

            for i in range(2):  # train twice as dicriminator is trained twice in each epoch
                noise2 = np.random.randn(batch_size, self.latent_dim)
                gen_loss = full_pass_model.train_on_batch(noise2, ones)
            history['generator_loss'].append(gen_loss)

            if epoch % log_period == 0:
                print(f'epoch {epoch + 1}/{n_epochs} - disc_loss: {history["discriminator_loss"][-1]}, disc_acc: {history["discriminator_accuracy"][-1]}, gen_loss: {history["generator_loss"][-1]}')

            if epoch % save_images_details['sample_period'] == 0:
                if 'image_shape' in save_images_details and save_images_details['image_shape'] is not None:
                    self.save_images(save_images_details['image_shape'], epoch)

        return history

    def save_images(self, image_shape, epoch):
        H, W = image_shape
        rows, cols = 5, 5
        noise = np.random.randn(rows * cols, self.latent_dim)
        imgs = self.generator_model.predict(noise)

        # Rescale images 0 - 1
        imgs = 0.5 * imgs + 0.5

        fig, axs = plt.subplots(rows, cols)
        idx = 0
        for i in range(rows):
            for j in range(cols):
                axs[i, j].imshow(imgs[idx].reshape(H, W), cmap='gray')
                axs[i, j].axis('off')
                idx += 1

        if not os.path.exists(self._save_path):
            os.mkdir(self._save_path)
        fig.savefig(os.path.join(self._save_path, f'{epoch}.png'))
        plt.close()

    def _build_generator_model(self):
        i = Input((self.latent_dim, ))
        x = Dense(256, activation=LeakyReLU(alpha=0.2))(i)
        x = BatchNormalization(momentum=0.7)(x)
        x = Dense(512, activation=LeakyReLU(alpha=0.2))(x)
        x = BatchNormalization(momentum=0.7)(x)
        x = Dense(1024, activation=LeakyReLU(alpha=0.2))(x)
        x = BatchNormalization(momentum=0.7)(x)
        x = Dense(self.D, activation='tanh')(x)
        return Model(i, x)

    def _build_discriminator_model(self):
        i = Input((self.D, ))
        x = Dense(512, activation=LeakyReLU(alpha=0.2))(i)
        x = Dense(256, activation=LeakyReLU(alpha=0.2))(x)
        x = Dense(1, activation='sigmoid')(x)
        return Model(i, x)

    def generate_fake(self):
        x = np.random.randn(self.D)
        return self.generator_model.redict(x)

    def plot_history(self, history):
        plt.figure(figsize=(10, 20))
        plt.subplot(1, 2, 1)
        plt.title('Loss')
        plt.plot(history['discriminator_loss'], label='discriminator loss')
        plt.plot(history['generator_loss'], label='generator loss')
        plt.legend()
        plt.subplot(1, 2, 2)
        plt.title('Accuracy')
        plt.plot(history['discriminator_accuracy'], label='discriminator accuracy')
        plt.legend()
        plt.show()
