from keras.layers import Input, Dense, Lambda
from keras.engine.topology import Layer
from keras.models import Model
from keras import backend as K
import tensorflow as tf
from tensorflow.contrib.distributions import MultivariateNormalDiag
from keras import metrics


# A class implementing the custom loss layer for Bernolli input,
# The call() method defines the operation.
# See https://keras.io/layers/writing-your-own-keras-layers/
class BernoVAE(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(BernoVAE, self).__init__(**kwargs)

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


# Create an and train an VAE model. Return the VAE, encoder, and decoder
def train_vae_berno(x_train, x_test, input_dim, hidden_dim=300, latent_dim=200, epsilon_std=1,
                    epochs=10, batch_size=100):
    # Implement the reparameterisation trick
    def sampling(args):
        z_mu, z_logsigma2 = args
        epsilon = K.random_normal(shape=(latent_dim,), mean=0., stddev=epsilon_std)
        return z_mu + K.exp(z_logsigma2 / 2) * epsilon

    x = Input(shape=(input_dim,))
    h = Dense(hidden_dim, activation='relu')(x)   # 1st hidden layer
    h = Dense(hidden_dim, activation='relu')(h)   # 2nd hidden layer
    z_mean = Dense(latent_dim)(h)                       # Use default linear activation
    z_logvar = Dense(latent_dim)(h)                     # Use default linear activation

    # Sampling layer. "output_shape" isn't necessary with the TensorFlow backend
    # Lambda wraps the function sampling() as a layer object
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_logvar])

    # We instantiate these layers separately so as to reuse them later
    # It is important to use 'sigmoid' as activation as vae_loss() in BernoVAE
    # uses binary_crossentropy as metrics.
    dec_h1 = Dense(hidden_dim, activation='relu')
    dec_mean = Dense(input_dim, activation='sigmoid')
    h1_decoded = dec_h1(z)                              # 1st hidden layer output of decoder
    x_decoded_mean = dec_mean(h1_decoded)               # Decoder output in original dim

    # Create a VAE model, using the BernoVAE.vae_loss() as the loss function
    # Passing loss=None to compile() causes the custom loss (vae_loss) to be used
    y = BernoVAE()([x, x_decoded_mean, z_logvar, z_mean])
    vae = Model(inputs=x, outputs=y)
    vae.compile(optimizer='adam', loss=None)

    # Train the VAE. No labels are needed as loss=None has been passed to vae.compile()
    vae.fit(x=x_train, y=None, shuffle=True, epochs=epochs, batch_size=batch_size,
            validation_data=(x_test, x_test))

    # Build an encoder that transforms input vector x vector z in the latent space
    encoder = Model(x, z_mean)

    # Build a decoder that transformes the encoded vector back to x
    decoder_input = Input(shape=(latent_dim,))
    _h1_decoded = dec_h1(decoder_input)
    _x_decoded_mean = dec_mean(_h1_decoded)
    decoder = Model(decoder_input, _x_decoded_mean)

    # Return VAE model, encoder and decoder
    return vae, encoder, decoder


# A class implementing the custom loss layer for Gaussian input,
# The call() method defines the operation.
# See https://keras.io/layers/writing-your-own-keras-layers/
class GaussVAE(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GaussVAE, self).__init__(**kwargs)

    def vae_loss(self, x, x_dec_mean, x_dec_lvar, zlogvar, zmean):
        diag_stdev = tf.exp(x_dec_lvar/2)
        dist = MultivariateNormalDiag(x_dec_mean, diag_stdev)
        gauss_loss = - dist.log_prob(x)       # Eq. 12 of Kingma (2014)
        kl_loss = - 0.5 * K.sum(1 + zlogvar - K.square(zmean) - K.exp(zlogvar), axis=-1)
        return K.mean(gauss_loss + kl_loss)   # Eq. 10 of Kingma (2014)

    # Define the operation of the custom layer in the call() method
    def call(self, inputs):
        x = inputs[0]
        x_dec_mean = inputs[1]
        x_dec_lvar = inputs[2]
        zlogvar = inputs[3]
        zmean = inputs[4]
        loss = self.vae_loss(x, x_dec_mean, x_dec_lvar, zlogvar, zmean)
        self.add_loss(loss, inputs=inputs)      # add_loss() is a function in keras.layers.Layer
        return x


# Create an and train an VAE model. Return the VAE, encoder, and decoder
def train_vae_gauss(x_train, x_test, input_dim, hidden_dim=300, latent_dim=200,
                    epsilon_std=1, epochs=10, batch_size=100):
    # Implement the reparameterisation trick
    def sampling(args):
        z_mu, z_log_sigma2 = args
        epsilon = K.random_normal(shape=(latent_dim,), mean=0., stddev=epsilon_std)
        return z_mu + K.exp(z_log_sigma2 / 2) * epsilon         # mu + sigma * epsilon

    # Define the encoder
    x = Input(shape=(input_dim, ))
    h = Dense(hidden_dim, activation='relu')(x)   # 1st Hidden layer
    h = Dense(hidden_dim, activation='relu')(h)   # 2nd Hidden layer
    z_mean = Dense(latent_dim)(h)                       # Use default linear activation
    z_log_var = Dense(latent_dim)(h)                    # Use default linear activation

    # Sampling layer. "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    # We instantiate these layers separately so as to reuse them later
    # It is important to use 'linear' as activation as vae_loss() in GaussVAE
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
    x_decoded_lvar = decoder_lvar(h_decoded[-1])               # \log\sigma_\theta^2

    # Create a VAE model, using the GaussVAE.vae_loss() as the loss function
    # Passing loss=None to compile() causes the custom loss (vae_loss) to be used
    y = GaussVAE()([x, x_decoded_mean, x_decoded_lvar, z_log_var, z_mean])
    vae = Model(inputs=x, outputs=y)
    vae.compile(optimizer='adam', loss=None)

    # Train the VAE. No labels as loss=None has been passed to vae.compile()
    vae.fit(x=x_train, y=None, shuffle=True, epochs=epochs, batch_size=batch_size,
            validation_data=(x_test, x_test))

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


