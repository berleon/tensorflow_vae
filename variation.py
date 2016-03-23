from tempfile import mkdtemp
import tensorflow as tf
import numpy as np
import keras.optimizers
import keras.models
from keras.backend import tensorflow_backend
from keras.layers import Dense, Activation, Reshape
from keras.optimizers import Adam
from keras.datasets import mnist

def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        it = iterable[ndx:min(ndx + n, l)]
        if len(it) == n:
            yield it

class VAE():
    def __init__(self, encoder, decoder):
        self.x = tf.placeholder(tf.float32, name='input')
        self.latent_shape = (encoder.output_shape[0], encoder.output_shape[1] // 2)
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = self.latent_shape[0]

        assert None not in self.latent_shape, "All dimensions must be known"
        encoded = tf.reshape(encoder(self.x), (self.batch_size, 2, self.latent_shape[1]))
        self.mu, self.log_sigma = encoded[:, 0, :], encoded[:, 1, :]
        self.mu = tf.reshape(self.mu, self.latent_shape)
        self.log_sigma = tf.reshape(self.log_sigma, self.latent_shape)

        self.eps = tf.random_normal(self.latent_shape,
                                    mean=0.0, stddev=1.0, name="eps")
        self.z = self.mu + tf.exp(self.log_sigma) * self.eps

        decoded = decoder(self.z)
        decoder_shape = decoder.output_shape
        if len(decoder_shape) == 2:
            decoded = tf.reshape(decoded, (self.batch_size, decoder_shape[1] // 2, 1, 2))
        else:
            assert decoder_shape[-1] == 2

        self.x_hat_mu, self.x_hat_log_sigma = decoded[:, :, :, 0], decoded[:, :, :, 1]
        self.x_hat_mu = tf.reshape(self.x_hat_mu, (self.batch_size, decoder_shape[1] // 2))
        self.x_hat_log_sigma = tf.reshape(self.x_hat_log_sigma, (self.batch_size, decoder_shape[1] // 2))

        self.params = encoder.trainable_weights + decoder.trainable_weights

        self.latent_loss = -0.5 * tf.reduce_mean(1 + self.log_sigma - self.mu**2 - tf.exp(self.log_sigma))
        self.reconstruction_loss = -tf.reduce_mean(((self.x_hat_mu - self.x)**2) / (2 * tf.exp(self.x_hat_log_sigma)))

        self.loss = self.latent_loss + self.reconstruction_loss

    def compile(self, optimizer):
        optimizer = keras.optimizers.get(optimizer)
        params = self.encoder.trainable_weights  + self.decoder.trainable_weights
        regularizers = self.encoder.regularizers  + self.decoder.regularizers
        constraints = self.encoder.constraints  + self.decoder.constraints
        updates = self.encoder.updates  + self.decoder.updates

        updates += optimizer.get_updates(params, constraints, self.loss)
        loss = self.loss
        for r in regularizers:
            loss += r(loss)
        self.train_loss = loss

        with tf.control_dependencies([self.train_loss]):
            self.train_updates = [tf.assign(p, new_p) for (p, new_p) in updates]

    def fit_batch(self, X, session):
        updated = session.run([self.train_loss] + self.train_updates, feed_dict={self.x: X})
        return updated[0]

    def fit(self, X, num_epochs=1):
        session = tensorflow_backend._get_session()
        writer_file = '/tmp/tmp0UWgeI'#mkdtemp()
        print(writer_file)
        writer = tf.train.SummaryWriter(writer_file, session.graph_def)
        for batch_idx in range(num_epochs):
            errors = []
            for x in batch(X, self.batch_size):
                errors.append(self.fit_batch(x, session))

            print('({}) Epoch error: {}'.format(batch_idx, np.mean(errors)))

    def reconstruct(self, X):
        session = tensorflow_backend._get_session()
        return session.run([self.x_hat_mu, self.x_hat_log_sigma], feed_dict={self.x: X})

    def encode(self, X):
        session = tensorflow_backend._get_session()
        return session.run([self.mu, self.log_sigma], feed_dict={self.x: X})

    def generate(self, Z=None):
        if Z is None:
            Z = np.random.normal(0, 1, self.latent_shape)

        session = tensorflow_backend._get_session()
        return session.run([self.x_hat_mu, self.x_hat_log_sigma], feed_dict={self.z: Z})


if __name__ == '__main__':
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    print(X_train.shape)

    z_dim = 10

    encoder = keras.models.Sequential()
    encoder.add(Dense(50, batch_input_shape=(64, 28 * 28)))
    encoder.add(Activation('tanh'))

    encoder.add(Dense(z_dim * 2, init='uniform'))
    encoder.add(Activation('tanh'))

    decoder = keras.models.Sequential()
    decoder.add(Dense(60, batch_input_shape=(64, z_dim)))
    decoder.add(Activation('tanh'))

    decoder.add(Dense(28 * 28 * 2))
    decoder.add(Activation('tanh'))

    optimizer = Adam()

    vae = VAE(encoder, decoder)
    vae.compile(optimizer)

    X_in = X_train.reshape((-1, 28 * 28)).astype(np.float32) / 255.
    print(X_in.shape)

    vae.fit(X_in * 2 - 1, num_epochs=10)
