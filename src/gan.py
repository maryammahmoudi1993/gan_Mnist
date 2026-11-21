"""Helpers shared by the notebooks: data, the two networks, the training loop and figures."""

import os
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")     # keep TensorFlow start-up messages quiet
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")    # avoid tiny run-to-run rounding differences

import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

SEED = 42
NOISE_DIM = 100
ASSETS = Path(__file__).resolve().parents[1] / "assets"


def set_seed(seed=SEED):
    """Fix every source of randomness we use, so reruns give the same numbers on a CPU."""
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def save_figure(fig, name):
    ASSETS.mkdir(exist_ok=True)
    fig.savefig(ASSETS / name, dpi=110, bbox_inches="tight")


# ---------------------------------------------------------------- data
def load_mnist():
    """The 60,000 training pictures as float32 numbers between 0 (white) and 1 (black), shape (60000, 28, 28, 1)."""
    (x_train, _), _ = keras.datasets.mnist.load_data()
    return (x_train.astype("float32") / 255.0)[..., None]


# ---------------------------------------------------------------- the two networks for digits
def build_generator(noise_dim=NOISE_DIM, width=64):
    """Noise (a list of 100 random numbers) -> a 28 x 28 picture with values between 0 and 1."""
    return keras.Sequential([
        keras.layers.Input((noise_dim,)),
        keras.layers.Dense(width * 7 * 7),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Reshape((7, 7, width)),
        keras.layers.Conv2DTranspose(width, 4, strides=2, padding="same"),      # 7 x 7 -> 14 x 14
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Conv2DTranspose(width, 4, strides=2, padding="same"),      # 14 x 14 -> 28 x 28
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Conv2D(1, 7, padding="same", activation="sigmoid"),
    ], name="generator")


def build_discriminator(width=64):
    """A picture -> one number between 0 (fake) and 1 (real)."""
    return keras.Sequential([
        keras.layers.Input((28, 28, 1)),
        keras.layers.Conv2D(width, 3, strides=2, padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Dropout(0.4),
        keras.layers.Conv2D(width, 3, strides=2, padding="same"),
        keras.layers.LeakyReLU(negative_slope=0.2),
        keras.layers.Dropout(0.4),
        keras.layers.Flatten(),
        keras.layers.Dense(1, activation="sigmoid"),
    ], name="discriminator")


# ---------------------------------------------------------------- a toy version: numbers instead of pictures
TOY_MEAN, TOY_STD = 3.0, 0.6


def toy_real_data(n=20000, seed=SEED):
    """The "real" data of the toy game: numbers from a bell curve with mean 3 and standard deviation 0.6."""
    return np.random.default_rng(seed).normal(TOY_MEAN, TOY_STD, size=(n, 1)).astype("float32")


def build_toy_generator():
    """One random number in, one number out."""
    return keras.Sequential([keras.layers.Input((1,)), keras.layers.Dense(16, activation="relu"),
                             keras.layers.Dense(1)], name="toy_generator")


def build_toy_discriminator():
    """One number in, a score between 0 (fake) and 1 (real) out."""
    return keras.Sequential([keras.layers.Input((1,)), keras.layers.Dense(16, activation="relu"),
                             keras.layers.Dense(1, activation="sigmoid")], name="toy_discriminator")


# ---------------------------------------------------------------- the game
def train_gan(generator, discriminator, data, steps, batch_size=128, snapshot_steps=(), noise_dim=None,
              fixed_noise=None, seed=SEED, learning_rate=2e-4):
    """Train the two networks against each other for ``steps`` steps.

    In every step: (1) the discriminator sees ``batch_size`` real and ``batch_size`` fake pictures and learns to tell them apart;
    (2) the generator makes new fakes and is rewarded when the discriminator calls them real.
    Call ``set_seed()`` before building the networks (their start weights are drawn then); the same seed is set again here for the noise and the dropout.
    ``data`` is an array of real examples. Returns (history, snapshots): the losses and accuracies of every step and the
    generator's output for ``fixed_noise`` at the steps listed in ``snapshot_steps`` (0 means before training).
    """
    set_seed(seed)
    noise_dim = noise_dim or generator.input_shape[1]
    bce = keras.losses.BinaryCrossentropy()
    g_opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    d_opt = keras.optimizers.Adam(learning_rate=learning_rate, beta_1=0.5)
    rng = np.random.default_rng(seed)
    data = tf.constant(data)

    @tf.function
    def step(real, noise_a, noise_b):
        ones, zeros = tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))
        with tf.GradientTape() as tape:                                    # 1) discriminator
            fake = generator(noise_a, training=True)
            p_real, p_fake = discriminator(real, training=True), discriminator(fake, training=True)
            d_loss = 0.5 * (bce(ones, p_real) + bce(zeros, p_fake))
        grads = tape.gradient(d_loss, discriminator.trainable_variables)
        d_opt.apply_gradients(zip(grads, discriminator.trainable_variables))
        with tf.GradientTape() as tape:                                    # 2) generator
            fake = generator(noise_b, training=True)
            g_loss = bce(ones, discriminator(fake, training=True))         # "please call my fakes real"
        grads = tape.gradient(g_loss, generator.trainable_variables)
        g_opt.apply_gradients(zip(grads, generator.trainable_variables))
        acc_real = tf.reduce_mean(tf.cast(p_real > 0.5, "float32"))
        acc_fake = tf.reduce_mean(tf.cast(p_fake < 0.5, "float32"))
        return d_loss, g_loss, acc_real, acc_fake

    history = {"d_loss": [], "g_loss": [], "d_acc_real": [], "d_acc_fake": []}
    snapshots = {}
    snapshot_steps = set(snapshot_steps)
    if 0 in snapshot_steps and fixed_noise is not None:
        snapshots[0] = generator(fixed_noise, training=False).numpy()
    for k in range(1, steps + 1):
        index = rng.integers(0, len(data), batch_size)
        noise_a = rng.standard_normal((batch_size, noise_dim)).astype("float32")
        noise_b = rng.standard_normal((batch_size, noise_dim)).astype("float32")
        values = step(tf.gather(data, index), noise_a, noise_b)
        for name, value in zip(history, values):
            history[name].append(float(value))
        if k in snapshot_steps and fixed_noise is not None:
            snapshots[k] = generator(fixed_noise, training=False).numpy()
    return history, snapshots


# ---------------------------------------------------------------- figures
def plot_grid(images, rows, cols, title=None):
    """A grid of pictures (shape (n, 28, 28, 1))."""
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 0.9, rows * 0.9 + (0.4 if title else 0)))
    for ax, image in zip(np.ravel(axes), images):
        ax.imshow(image[:, :, 0], cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
    if title:
        fig.suptitle(title, fontsize=10)
    return fig


def plot_game():
    """The game between the two networks, drawn with boxes."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.axis("off")
    boxes = {"noise": (0.0, 0.8, "random noise\n(100 numbers)"), "G": (2.0, 0.8, "generator\n(the forger)"),
             "fake": (4.0, 0.8, "fake picture"), "real": (4.0, -0.2, "real picture\n(from MNIST)"),
             "D": (6.2, 0.3, "discriminator\n(the detective)"), "out": (8.4, 0.3, "\"real\" or \"fake\"?\none number 0 to 1")}
    for key, (x, y, text) in boxes.items():
        color = {"G": "#fde7d9", "D": "#dbe9f7"}.get(key, "#f2f2f2")
        ax.text(x, y, text, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=0.5", fc=color, ec="#555555"))
    arrows = [("noise", "G"), ("G", "fake"), ("fake", "D"), ("real", "D"), ("D", "out")]
    for a, b in arrows:
        (x1, y1, _), (x2, y2, _) = boxes[a], boxes[b]
        ax.annotate("", xy=(x2 - 0.75, y2 + (y1 - y2) * 0.2), xytext=(x1 + 0.75, y1),
                    arrowprops=dict(arrowstyle="->", color="#555555"))
    ax.text(6.2, -0.75, "the detective learns to tell them apart;\nthe forger learns from the detective's answers", ha="center", fontsize=8, color="#444444")
    ax.set_xlim(-1, 9.6); ax.set_ylim(-1.2, 1.4)
    return fig


def plot_toy_progress(real, snapshots):
    """Histograms of the numbers made by the generator at several moments, against the real data."""
    steps = sorted(snapshots)
    fig, axes = plt.subplots(1, len(steps), figsize=(2.6 * len(steps), 2.6), sharey=True, sharex=True)
    for ax, k in zip(axes, steps):
        ax.hist(real, bins=60, range=(-2, 6), density=True, color="tab:green", alpha=.45, label="real")
        ax.hist(snapshots[k], bins=60, range=(-2, 6), density=True, color="tab:red", alpha=.55, label="generated")
        ax.set_title(f"step {k}", fontsize=9)
        ax.set_xlabel("value")
    axes[0].set_ylabel("density"); axes[0].legend(fontsize=7)
    return fig


def plot_progress(snapshots, columns=8):
    """One row per moment of training: the generator's output for the same noise."""
    steps = sorted(snapshots)
    fig, axes = plt.subplots(len(steps), columns, figsize=(columns * 0.95, len(steps) * 0.95 + 0.3))
    for r, k in enumerate(steps):
        for c in range(columns):
            axes[r, c].imshow(snapshots[k][c, :, :, 0], cmap="gray_r", vmin=0, vmax=1)
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(f"step {k}", rotation=0, ha="right", va="center", fontsize=8)
    return fig
