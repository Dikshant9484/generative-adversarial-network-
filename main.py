import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt

(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
x_train = x_train.astype("float32") / 127.5 - 1.0

BUFFER_SIZE = 60000
BATCH_SIZE = 128
LATENT_DIM = 100

dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def build_generator():
    model = tf.keras.Sequential([
        layers.Input(shape=(LATENT_DIM,)),
        layers.Dense(8*8*256, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, 3, strides=1, padding="same", activation="tanh", use_bias=False)
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        layers.Input(shape=(32,32,3)),
        layers.Conv2D(64, 4, strides=2, padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, 4, strides=2, padding="same"),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

generator = build_generator()
discriminator = build_discriminator()

loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def d_loss(real, fake):
    real_loss = loss_fn(tf.ones_like(real), real)
    fake_loss = loss_fn(tf.zeros_like(fake), fake)
    return real_loss + fake_loss

def g_loss(fake):
    return loss_fn(tf.ones_like(fake), fake)

g_opt = tf.keras.optimizers.Adam(1e-4)
d_opt = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        generated = generator(noise, training=True)
        real_out = discriminator(images, training=True)
        fake_out = discriminator(generated, training=True)
        gen_loss = g_loss(fake_out)
        disc_loss = d_loss(real_out, fake_out)

    g_grads = g_tape.gradient(gen_loss, generator.trainable_variables)
    d_grads = d_tape.gradient(disc_loss, discriminator.trainable_variables)

    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))
    d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

EPOCHS = 50

for epoch in range(EPOCHS):
    for batch in dataset:
        if batch.shape[0] == BATCH_SIZE:
            train_step(batch)
    print("Epoch", epoch+1, "completed")

noise = tf.random.normal([16, LATENT_DIM])
generated = generator(noise, training=False)

plt.figure(figsize=(4,4))
for i in range(16):
    img = (generated[i] + 1) / 2
    plt.subplot(4,4,i+1)
    plt.imshow(img)
    plt.axis("off")

plt.show()
