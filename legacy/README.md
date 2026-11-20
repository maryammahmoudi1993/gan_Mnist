# Generative Adversarial Networks (GAN) for Mnist
generate handwriting with GAN using mnist dataset
# written by: Maryam Mahmoudi 26 Nov 2022

This code is an implementation of a Generative Adversarial Network (GAN) using TensorFlow.

GANs are a type of neural network that can generate new data that is similar to a training dataset. They consist of two neural networks: a generator and a discriminator. The generator creates new samples of data, while the discriminator evaluates the samples for authenticity.

In this code, the GAN is used to generate new images of handwritten digits. The generator takes a random noise vector as input and generates an image that resembles a handwritten digit. The discriminator takes an image as input and outputs a binary classification indicating whether the image is real or fake.

# Dependencies
os - Miscellaneous operating system interfaces.
numpy - Package for scientific computing with Python.
tensorflow - An open source machine learning framework.
matplotlib - A plotting library for the Python programming language.
# Variables
batch_size - The number of samples to use in each training batch.
in_shape - The shape of the input images.
noise_dim - The dimension of the input noise vector for the generator.
# Functions
generator(data) - Function that defines the generator neural network.
discriminator() - Function that defines the discriminator neural network.
noise_data(noise_dim, batch_size) - Function that generates a batch of random noise vectors of dimension noise_dim.
fake_data(generator, noise_dim, batch_size) - Function that generates a batch of fake images using the generator.
gan(g_model, d_model) - Function that defines the GAN model using the generator and discriminator.
real_data() - Function that loads the MNIST dataset and normalizes the images.
real_sampling(real_data, batch_size) - Function that selects a batch of real images from the dataset.
save_plot(examples, epoch, n=10) - Function that saves a plot of generated images.
summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100) - Function that evaluates the generator and discriminator on a batch of real and fake images and saves a plot of generated images.
train(g_model, d_model, gan, re_data, noise_dim, n_epochs=100, n_batch=256) - Function that trains the GAN.
# Usage
To use this code, first ensure that the dependencies are installed. Then, import the code into a Python script or notebook.
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Variables
batch_size = 32
in_shape = (28,28,1)
noise_dim = 100

# Define Generator
def generator(data):
    ...

# Define Discriminator
def discriminator():
    ...

# Generate Noise data
def noise_data(noise_dim, batch_size):
    ...

# Generate fake data
def fake_data(generator, noise_dim, batch_size):
    ...

# Define Gan model
def gan(g_model, d_model):
    ...

# Load Real data
def real_data():
    ...

# Sampling from real data
def real_sampling(real_data, batch_size):
    ...

# Define trainer (like fit operator in Deep)
def train(g_model, d_model, gan, re_data, noise_dim, n_epochs=100, n_batch=256):
    ...

# Generate fake images and save plot
X,Y = fake_data(generator=generator, noise_dim=100, batch_size=batch_size)
save_plot(X, 0)

# Train GAN
train(g_model, d_model
