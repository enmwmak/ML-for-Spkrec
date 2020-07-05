# Implemetation of Bernolli variational autoencoder with API similar to sklearn
# Author: M.W. Mak
# Date: Dec. 2018

from keras.layers import Input, Dense, Lambda
from keras.engine.topology import Layer
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalDiag
from keras import metrics
import numpy as np

class Transform:
    def __init__(self):
        self.trained = False

    def fit(self, X, spk_ids):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def fit_transform(self, X, spk_ids=None):
        if self.trained:
            return self.transform(X)
        else:
            return self.fit(X, spk_ids).transform(X)

    def check_if_model_exist(self):
        if self.trained:
            raise ValueError('a trained model already exist.')
        else:
            self.trained = True


# VAE with Bernolli distributed input vectors
class BernolliVAE(Transform):
    def __init__(self, input_dim=512, hidden_dim=300, latent_dim=200, epsilon_std=1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        self.vae, self.encoder, self.decoder = self.create_berno_vae(input_dim, hidden_dim, latent_dim, epsilon_std)

    # Create a Keras DNN that implements a Bernolli VAE. Do not train it yet.
    # Return a DNN comprising encoder and decoder (VAE). Also return the encoder and the decoder of that VAE
    def create_berno_vae(self, input_dim, hidden_dim, latent_dim, epsilon_std):

        # Implement the reparameterisation trick
        def sampling(args):
            z_mu, z_logsigma2 = args
            epsilon = K.random_normal(shape=(latent_dim,), mean=0., stddev=epsilon_std)
            return z_mu + K.exp(z_logsigma2 / 2) * epsilon

        # Define the encoder
        x = Input(shape=(input_dim, ))
        h = Dense(hidden_dim, activation='relu')(x)   # 1st Hidden layer
        h = Dense(hidden_dim, activation='relu')(h)   # 2nd Hidden layer
        z_mean = Dense(latent_dim)(h)                       # Use default linear activation
        z_log_var = Dense(latent_dim)(h)                    # Use default linear activation

        # Sampling layer. "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # We instantiate these layers separately so as to reuse them later
        # It is important to use 'linear' as activation as vae_loss() in GaussCustomLayer
        # uses logGaussian as metrics.
        n_decoder_layers = 1                    # 1 hidden layer achieves the best performance on MNIST data
        decoder_h = []
        h_decoded = []
        for i in range(n_decoder_layers):
            decoder_h.append(Dense(hidden_dim, activation='relu'))               
        decoder_mean = Dense(input_dim, activation='linear')
        decoder_lvar = Dense(input_dim, activation='linear')

        h_decoded.append(decoder_h[0](z))                            # Output of 1st hidden layer
        for i in range(1, n_decoder_layers):
            h_decoded.append(decoder_h[i](h_decoded[i-1]))
        x_decoded_mean = decoder_mean(h_decoded[-1])               # \mu_\theta(z)

        # Create a VAE model, using the BernoCustomLayer.vae_loss() as the loss function
        # Passing loss=None to compile() causes the custom loss (vae_loss) to be used
        y = BernoCustomLayer()([x, x_decoded_mean, z_log_var, z_mean])
        vae = Model(inputs=x, outputs=y)
        vae.compile(optimizer='adam', loss=None)

        # Build a model to project inputs on the latent space (encoder)
        encoder = Model(inputs=x, outputs=z_mean)

        # Build a decoder that transformes the encoded vector back to x
        decoder_input = Input(shape=(latent_dim,))
        _h_decoded = []
        _h_decoded.append(decoder_h[0](decoder_input))
        for i in range(1, n_decoder_layers):
            _h_decoded.append(decoder_h[i](_h_decoded[i-1]))
        _x_decoded_mean = decoder_mean(_h_decoded[-1])
        decoder = Model(inputs=decoder_input, outputs=_x_decoded_mean)

        # Return VAE model and Encoder
        return vae, encoder, decoder

    # Train the VAD using x_train and validate it using x_test
    def fit(self, x_train, x_test, epochs=10, batch_size=100):
        self.check_if_model_exist()

        # Train the VAE. No labels as loss=None has been passed to vae.compile()
        self.vae.fit(x=x_train, y=None, shuffle=True, epochs=epochs, batch_size=batch_size,
                     validation_data=(x_test, x_test))
        return self

    # Perform end-to-end transform of X using the VAE
    def transform(self, X, batch_size=100):
        return self.vae.predict(X, batch_size=batch_size)

    # Encode input X using the encoder
    def encode(self, X, batch_size=100):
        self.encoder.predict(X, batch_size=batch_size)    


# A class implementing the custom loss layer for Bernolli input,
# The call() method defines the operation.
# See https://keras.io/layers/writing-your-own-keras-layers/
class BernoCustomLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(BernoCustomLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_dec_mean, zlogvar, zmean):
        input_dim = int(x.shape[1])
        xent_loss = input_dim * metrics.binary_crossentropy(x, x_dec_mean)  # Eq. 11 of Kingma (2014)
        kl_loss = - 0.5 * K.sum(1 + zlogvar - K.square(zmean) - K.exp(zlogvar), axis=-1)  # Sum over the last axis
        return K.mean(xent_loss + kl_loss)   # Eq. 10 of Kingma (2014)

    def call(self, inputs):
        x = inputs[0]
        x_dec_mean = inputs[1]
        zlogvar = inputs[2]
        zmean = inputs[3]        
        loss = self.vae_loss(x, x_dec_mean, zlogvar, zmean)
        self.add_loss(loss, inputs=inputs)      # add_loss() is a function in keras.layers.Layer
        return x


# Main function for testing the GaussianVAE class
def main():
    BernolliVAE(input_dim=512, hidden_dim=300, latent_dim=200, epsilon_std=1)


# Entry point of this script. It will set up Tensorflow, GPU and call the main function
if __name__ == '__main__':
    # Use 1/3 of the GPU memory so that the GPU can be shared by multiple users
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    np.random.seed(1)

    # Ignore warnings
    def warn(*args, **kwargs):
        pass
    import warnings 
    warnings.warn = warn
    main()