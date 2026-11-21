"""Quick checks that the helpers work. They do not train anything for long."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import gan  # noqa: E402


def test_data_range_and_shape():
    x = gan.load_mnist()
    assert x.shape == (60000, 28, 28, 1) and x.min() == 0.0 and x.max() == 1.0


def test_networks_have_expected_shapes():
    noise = np.zeros((3, gan.NOISE_DIM), dtype="float32")
    fake = gan.build_generator(width=8)(noise)
    assert fake.shape == (3, 28, 28, 1) and 0.0 <= float(fake.numpy().min()) and float(fake.numpy().max()) <= 1.0
    assert gan.build_discriminator(width=8)(fake).shape == (3, 1)


def test_training_runs_and_is_reproducible():
    data = gan.load_mnist()[:512]
    noise = np.random.default_rng(0).standard_normal((4, gan.NOISE_DIM)).astype("float32")
    results = []
    for _ in range(2):
        gan.set_seed()                                   # the start weights are drawn when the networks are built
        generator, discriminator = gan.build_generator(width=8), gan.build_discriminator(width=8)
        history, snapshots = gan.train_gan(generator, discriminator, data, steps=5, batch_size=16,
                                           snapshot_steps=(0, 5), fixed_noise=noise)
        results.append((history["g_loss"][-1], snapshots[5].sum()))
    assert len(history["d_loss"]) == 5 and set(snapshots) == {0, 5} and results[0] == results[1]


def test_toy_game_moves_the_generator_towards_the_real_numbers():
    real = gan.toy_real_data(n=4000)
    noise = np.random.default_rng(1).standard_normal((2000, 1)).astype("float32")
    generator, discriminator = gan.build_toy_generator(), gan.build_toy_discriminator()
    _, snapshots = gan.train_gan(generator, discriminator, real, steps=1500, batch_size=128,
                                 snapshot_steps=(0, 1500), fixed_noise=noise, learning_rate=1e-3)
    before, after = abs(snapshots[0].mean() - gan.TOY_MEAN), abs(snapshots[1500].mean() - gan.TOY_MEAN)
    assert after < before and after < 1.0
