import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
import os
import gdown
from zipfile import ZipFile
from tqdm import tqdm

# === SETUP ===
if not os.path.isdir("celeba_gan"):
    os.makedirs("celeba_gan")

# Download dataset
url = "https://drive.google.com/uc?id=1O7m1010EJjLE5QxLZiM9Fpjs7Oj6e684"
output = "celeba_gan/data.zip"
if not os.path.exists(output):
    gdown.download(url, output, quiet=False)

# Extract ZIP
with ZipFile(output, "r") as zipobj:
    zipobj.extractall("celeba_gan")

# Load dataset
dataset = keras.preprocessing.image_dataset_from_directory(
    directory="celeba_gan",
    label_mode=None,
    image_size=(64, 64),
    batch_size=32,
    shuffle=True
).map(lambda x: x / 255.0)

# Visualize sample
for x in dataset.take(1):
    plt.axis("off")
    plt.imshow((x.numpy() * 255).astype("int32")[0])
    plt.show()

# === DISCRIMINATOR ===
discriminator = keras.Sequential([
    keras.Input(shape=(64, 64, 3)),
    layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(0.2),
    layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(0.2),
    layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(0.2),
    layers.Flatten(),
    layers.Dropout(0.2),
    layers.Dense(1, activation="sigmoid")
])
discriminator.summary()

# === GENERATOR ===
latent_dim = 128
generator = keras.Sequential([
    layers.Input(shape=(latent_dim,)),
    layers.Dense(8 * 8 * 128),
    layers.Reshape((8, 8, 128)),
    layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(0.2),
    layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(0.2),
    layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
    layers.LeakyReLU(0.2),
    layers.Conv2D(3, kernel_size=5, padding="same", activation="sigmoid")
])
generator.summary()

# === TRAINING SETUP ===
opt_gen = keras.optimizers.Adam(1e-4)
opt_disc = keras.optimizers.Adam(1e-4)
loss_fn = keras.losses.BinaryCrossentropy()
epochs = 10

# Folder for results
os.makedirs("generated_images", exist_ok=True)

# === TRAINING LOOP ===
for epoch in range(epochs):
    print(f"\nEpoch {epoch + 1}/{epochs}")
    for idx, real in enumerate(tqdm(dataset)):
        batch_size = real.shape[0]

        # === Generate Fake Images ===
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        fake = generator(random_latent_vectors)

        # === Train Discriminator ===
        with tf.GradientTape() as disc_tape:
            real_output = discriminator(real)
            fake_output = discriminator(fake)

            loss_real = loss_fn(tf.ones((batch_size, 1)), real_output)
            loss_fake = loss_fn(tf.zeros((batch_size, 1)), fake_output)
            loss_disc = (loss_real + loss_fake) / 2

        grads_disc = disc_tape.gradient(loss_disc, discriminator.trainable_weights)
        opt_disc.apply_gradients(zip(grads_disc, discriminator.trainable_weights))

        # === Train Generator ===
        random_latent_vectors = tf.random.normal(shape=(batch_size, latent_dim))
        with tf.GradientTape() as gen_tape:
            generated_images = generator(random_latent_vectors)
            fake_output = discriminator(generated_images)
            loss_gen = loss_fn(tf.ones((batch_size, 1)), fake_output)

        grads_gen = gen_tape.gradient(loss_gen, generator.trainable_weights)
        opt_gen.apply_gradients(zip(grads_gen, generator.trainable_weights))

        # === Save Generated Image Every 100 Steps ===
        if idx % 100 == 0:
            img = keras.preprocessing.image.array_to_img(generated_images[0])
            img.save(f"generated_images/generated_img_{epoch}_{idx}.png")
            print(f"Saved generated image: epoch {epoch}, step {idx}")

