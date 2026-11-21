# Legacy files

These are my original files from the first version of this project (26 Nov 2022, README updated Feb 2023), kept unchanged:
`gan01.py` (the script) and the first `README.md`.

What I notice today, from reading the code:

- **The training stops with an error at epoch 10.** At the first summary the script calls
  `summarize_performance(i, g_model, d_model, re_data, noise_data)`: it passes the *function* `noise_data` where the
  *number* `noise_dim` is expected, so `np.random.randn(...)` fails. I never saw a result from the script beyond epoch 10.
- **Global variables hide mistakes.** `fake_data` generates images with the global `g_model` instead of the `generator`
  argument (and is even called with the *function* `generator`), and `train` uses the global `GAN` instead of its `gan` argument.
  It works only because the names happen to be the same.
- The models are created twice, the original `g_model`, `d_model` and `GAN` at the top and again at the bottom of the file.
- It writes a PNG and a `generator_model_*.h5` model file into the working directory every ten epochs (the model files do not belong in a repository).
- There is no random seed, no record of the losses and no way to see *how* training went; the README describes the code
  but shows no result, no picture and no discussion of why GANs are hard to train.
- 100 epochs of 234 steps each is a long run for a CPU; no smaller setting was offered.

The new notebooks in `../notebooks/` rebuild the project step by step: the idea of the game between two networks,
a toy GAN with a picture that makes it visible, and a small GAN for handwritten digits. Part 2 adds samples across training,
loss curves, instability and mode collapse.
