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
    n_node = 128 * 7 * 7
    net = tf.keras.models.Sequential([
                                    tf.keras.layers.Dense(n_node, input_dim = data),
                                    tf.keras.layers.LeakyReLU(alpha=0.2),
                                    tf.keras.layers.Reshape((7,7,128)),
                                    tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
                                    tf.keras.layers.LeakyReLU(alpha=0.2),
                                    tf.keras.layers.Conv2DTranspose(128, (4,4), strides=(2,2), padding='same'),
                                    tf.keras.layers.LeakyReLU(0.2),
                                    tf.keras.layers.Conv2D(1, (7,7), activation='sigmoid' , padding='same')    ])
    '''net.compile(optimizer='adam', loss='categorical_cross_entropy', metrics=['accuracy'])
    net.summary()'''
    return net
g_model = generator(noise_dim)
# Define Discriminator
def discriminator():
    
    net = tf.keras.models.Sequential([
                                    tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same', input_shape=in_shape),
                                    tf.keras.layers.LeakyReLU(alpha=0.2),
                                    tf.keras.layers.Dropout(0.4),
                                    tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), padding='same'),
                                    tf.keras.layers.LeakyReLU(alpha=0.2),
                                    tf.keras.layers.Dropout(0.4),
                                    tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(1, activation='sigmoid' )    ])
                                    
    opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    net.compile(optimizer=opt , loss='binary_crossentropy', metrics=['accuracy'])
    #net.summary()
    return net
d_model = discriminator()
# Generate Noise data
def noise_data(noise_dim, batch_size):  # noise_dim = dimention of noise(100) , batch_size = size of batch(32)
    x_input = np.random.randn(noise_dim * batch_size)  # 3200 random numbers
    x_input = x_input.reshape(batch_size, noise_dim)   # 32 * [100-length]
    return x_input
# Generate fake data
def fake_data(generator, noise_dim, batch_size): 
    x_input = noise_data(noise_dim, batch_size)
    X = g_model.predict(x_input)
    Y = np.zeros((batch_size,1))
    return X, Y
X,Y = fake_data(generator=generator, noise_dim=100, batch_size=batch_size)
# Define Gan model
def gan(g_model, d_model):
    d_model.trainable = False  # in order not to train disciminator when generator is training
    model = tf.keras.models.Sequential([
        g_model,
        d_model
    ])
    opt =  tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])
    return model
GAN = gan(g_model=g_model, d_model=d_model)
# Load Real data
def real_data():
    (x_train,_),(_,_) = tf.keras.datasets.mnist.load_data()
    x = np.reshape(x_train, (len(x_train),28,28,1))
    x = x.astype('float32')
    x = x/255.0
    return x
re_data = real_data()
# Sampling from real data
def real_sampling(real_data, batch_size):
    ix = np.random.randint(0, re_data.shape[0], batch_size) # generate index in size of batch size (32) from 0 to the end of data (60000)
    x = real_data[ix] # return data coresponds to the indexes that is defined
    y = np.ones((batch_size,1)) # asign 1 label to the real data
    return x, y 
x_real, y_real = real_sampling(real_data=re_data, batch_size=batch_size)
def save_plot(examples, epoch, n=10):
	for i in range(n * n):
		plt.subplot(n, n, 1 + i)
		plt.axis('off')
		plt.imshow(examples[i, :, :, 0], cmap='gray_r')
	filename = 'generated_plot_e%03d.png' % (epoch+1)
	plt.savefig(filename)
	plt.close()
def summarize_performance(epoch, g_model, d_model, dataset, latent_dim, n_samples=100):
    X_real, y_real = real_sampling(dataset, n_samples)
    _, acc_real = d_model.evaluate(X_real, y_real, verbose=0)
    x_fake, y_fake = fake_data(g_model, latent_dim, n_samples)
    _, acc_fake = d_model.evaluate(x_fake, y_fake, verbose=0)
    print(f'>Accuracy real: {acc_real*100}, fake: {acc_fake*100}')
    save_plot(x_fake, epoch)
    filename = f'generator_model_{epoch + 1}.h5'
    g_model.save(filename)
# Define trainer (like fit operator in Deep)
def train(g_model, d_model, gan, re_data, noise_dim, n_epochs=100, n_batch=256):
    bat_per_epoch = int(re_data.shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epoch):
            # train discrimintaor
            x_real, y_real = real_sampling(real_data=re_data, batch_size=half_batch)
            x_fake, y_fake = fake_data(generator=g_model, noise_dim=noise_dim, batch_size=half_batch)
            X, y = np.vstack((x_real, x_fake)), np.vstack((y_real, y_fake))
            d_loss, _ = d_model.train_on_batch(X,y) # train discriminator just for one batch
            # train generator(total GAN)
            x_gan = noise_data(noise_dim=noise_dim, batch_size=n_batch) # to train Gan, we need to feed it noise data
            y_gan = np.ones((n_batch,1)) # we label Gan as 1 because we need to label it as true (opposite of disciminator)
            g_loss,_ = GAN.train_on_batch(x_gan,y_gan) # train gan model just for one batch
            # show results
            print(f">{i+1}, {j+1}/{bat_per_epoch}, d= {d_loss:.3f}, g={g_loss:.3f}")
            if (i+1) % 10 == 0 :
                summarize_performance(i, g_model, d_model, re_data, noise_data)

d_model = discriminator()
g_model = generator(noise_dim)
gan_model = gan(g_model, d_model)
dataset = real_data()
train(g_model, d_model, gan_model, dataset, noise_dim)