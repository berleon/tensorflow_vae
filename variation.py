

import tensorflow as tf
import numpy as np
import input_data


def weight_init(shape):
    return np.random.normal(0, 0.05, shape)


class VAE():
    def __init__(self, encoder, decoder, zdims, output_shape, input_shape):
        self.x = tf.placeholder(tf.float32)
        self.zdims = zdims
        self.input_shape = input_shape
        self.output_shape = output_shape

        e_shp = (zdims, output_shape[1])
        self.W_sigma = tf.Variable(weight_init(e_shp), name="W_esigma")
        self.W_mu = tf.Variable(weight_init(e_shp), name="W_emu")

        self.encoded = encoder(self.x)
        self.mu = tf.matmul(self.W_mu, self.encoded)
        self.log_sigma = tf.matmul(self.W_sigma, self.encoded)

        self.eps = tf.random_normal((output_shape[0], zdims),
                                    mean=0.0, stddev=1.0, name="eps")
        self.z = self.mu + tf.exp(self.log_sigma) * self.eps
        self.x_hat = decoder(self.z)

        self.params = [self.W_sigma, self.W_mu, self.W_d]


        self.prior = 0.5* tf.sum(1 + 2*log_sigma_encoder - mu_encoder**2 - T.exp(2*log_sigma_encoder))

    def compile(self):
        pass

    def fit(self, X, y):
        pass


if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    vae = VAE(lambda x: x, )
    vae.fit(mnist.images(), mnist.labels())
