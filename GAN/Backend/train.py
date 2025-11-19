import tensorflow as tf
from tensorflow.keras import layers, mixed_precision
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from tqdm.auto import tqdm  # pink progress bars

# -------------------------------------------
# ENABLE MIXED PRECISION (Apple Silicon)
# -------------------------------------------
mixed_precision.set_global_policy("mixed_float16")
print("Mixed precision:", mixed_precision.global_policy())

# -------------------------------------------
# LOAD CIFAR-10
# -------------------------------------------
(x_train, _), (_, _) = tf.keras.datasets.cifar10.load_data()
x_train = tf.image.resize(x_train, (256, 256)).numpy()
x_train = (x_train.astype("float32") - 127.5) / 127.5  # Normalize to [-1, 1]

BUFFER_SIZE = 60000
BATCH_SIZE = 256  # Fast on Apple M-series GPUs

dataset = (
    tf.data.Dataset.from_tensor_slices(x_train)
    .cache()
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.AUTOTUNE)
)

# -------------------------------------------
# GENERATOR (FAST)
# -------------------------------------------
def build_generator():
    inputs = layers.Input(shape=(100,))
    x = layers.Dense(4 * 4 * 512, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Reshape((4, 4, 512))(x)

    x = layers.Conv2DTranspose(256, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(128, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    outputs = layers.Conv2DTranspose(3, 3, activation="tanh", padding="same")(x)
    return tf.keras.Model(inputs, outputs)

# -------------------------------------------
# DISCRIMINATOR (FAST)
# -------------------------------------------
def build_discriminator():
    inputs = layers.Input(shape=(256, 256, 3))

    x = layers.Conv2D(32, 4, strides=2, padding="same")(inputs)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(64, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(128, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(256, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Conv2D(512, 4, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(1)(x)  # logits
    return tf.keras.Model(inputs, outputs)

generator = build_generator()
discriminator = build_discriminator()

# -------------------------------------------
# LOSS + OPTIMIZERS
# -------------------------------------------
loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
g_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
d_opt = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# -------------------------------------------
# TRAINING STEP
# -------------------------------------------
@tf.function
def train_step(real_images):
    noise = tf.random.normal([BATCH_SIZE, 100])

    # Train discriminator
    with tf.GradientTape() as tape:
        fake_images = generator(noise, training=True)
        real_out = discriminator(real_images, training=True)
        fake_out = discriminator(fake_images, training=True)

        d_loss_real = loss_fn(tf.ones_like(real_out), real_out)
        d_loss_fake = loss_fn(tf.zeros_like(fake_out), fake_out)
        d_loss = d_loss_real + d_loss_fake

    d_grads = tape.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

    # Train generator
    with tf.GradientTape() as tape:
        noise = tf.random.normal([BATCH_SIZE, 100])
        fake_images = generator(noise, training=True)
        fake_out = discriminator(fake_images, training=True)
        g_loss = loss_fn(tf.ones_like(fake_out), fake_out)

    g_grads = tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

    return g_loss, d_loss

# -------------------------------------------
# IMAGE & MODEL SAVING
# -------------------------------------------
seed = tf.random.normal([25, 100])

def save_images(epoch):
    preds = generator(seed, training=False)
    preds = tf.cast((preds + 1) / 2.0, tf.float32).numpy()

    fig = plt.figure(figsize=(5, 5))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(preds[i])
        plt.axis("off")

    os.makedirs("samples", exist_ok=True)
    plt.tight_layout()
    plt.savefig(f"samples/epoch_{epoch}.png")
    plt.close()

def save_models(epoch):
    os.makedirs("models", exist_ok=True)
    generator.save(f"models/generator_epoch_{epoch}.h5")
    discriminator.save(f"models/discriminator_epoch_{epoch}.h5")
    print(f"Saved milestone models at epoch {epoch}")

# -------------------------------------------
# TRAIN LOOP WITH ALWAYS-VISIBLE PINK PROGRESS BAR 🌸
# -------------------------------------------
EPOCHS = 10
milestones = [1, 5, 10]

for epoch in range(1, EPOCHS + 1):
    print(f"\n🌸 Epoch {epoch}/{EPOCHS}")

    progress = tqdm(
        dataset,
        desc=f"🌸 Training Epoch {epoch}",
        colour="magenta",
        file=sys.stdout,       # ensures visibility
        dynamic_ncols=True,    # adjusts to terminal width
        leave=False
    )

    for real_batch in progress:
        g_loss, d_loss = train_step(real_batch)
        progress.set_postfix({
            "G_loss": float(g_loss),
            "D_loss": float(d_loss)
        })

    print(f" ✔ Finished Epoch {epoch} | G Loss: {g_loss:.4f} | D Loss: {d_loss:.4f}")

    # ------------------------------
    # SAMPLE SAVING: every 10 epochs
    # ------------------------------
    if epoch % 10 == 0:
        save_images(epoch)

    # --------------------------------------
    # MODEL SAVING: only at milestone epochs
    # --------------------------------------
    if epoch in milestones:
        save_models(epoch)
        save_images(epoch)  # also save sample at milestone
