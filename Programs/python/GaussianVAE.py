# Implemetation of Gaussian variational autoencoder with API similar to sklearn
# Author: M.W. Mak
# Date: Dec. 2018

from keras.layers import Input, Dense, Lambda, Embedding, Concatenate, BatchNormalization, Dropout
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


# VAE with Gaussian distributed input vectors
class GaussianVAE(Transform):
    def __init__(self, input_dim=512, hidden_dim=300, latent_dim=200, epsilon_std=1, 
                isCenterloss=False, isCEloss=False, n_spks=None):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.epsilon_std = epsilon_std
        self.isCenterloss = isCenterloss
        self.n_spks = n_spks
        #if isCenterloss:
        #    self.vae, self.encoder, self.decoder = self.create_gvae_closs(input_dim, hidden_dim, latent_dim, n_spks, epsilon_std)
        #else:
        #    self.vae, self.encoder, self.decoder = self.create_gvae(input_dim, hidden_dim, latent_dim, n_spks, epsilon_std)
        self.vae, self.encoder, self.decoder = self.create_gvae(input_dim, hidden_dim, latent_dim, epsilon_std,
                                        isCenterloss=isCenterloss, isCEloss=isCEloss, n_spks=n_spks)


    # Create a DNN that implements a Gaussian VAE with optional center loss and cross-entropy loss. Do not train it yet.
    # Return a DNN comprising encoder and decoder (VAE). Also return the encoder and the decoder of that VAE
    def create_gvae(self, input_dim, hidden_dim, latent_dim, epsilon_std,
                    isCenterloss=False, isCEloss=False, n_spks=None):

        # Implement the reparameterisation trick
        def sampling(args):
            z_mu, z_log_sigma2 = args
            epsilon = K.random_normal(shape=(latent_dim,), mean=0., stddev=epsilon_std)
            return z_mu + K.exp(z_log_sigma2 / 2) * epsilon         # mu + sigma * epsilon

        # Define the encoder
        n_encoder_layers = 1
        x = Input(shape=(input_dim, ))
        h = x
        for i in range(n_encoder_layers):
            h = Dense(hidden_dim, activation='relu')(h)   # Hidden layer
            # h = Dropout(0.2)(h)                         # Using Dropout leads to poor performance  
        z_mean = Dense(latent_dim)(h)                     # Linear activation for mean
        z_log_var = Dense(latent_dim)(h)                  # Linear activation for log var

        # Sampling layer. "output_shape" isn't necessary with the TensorFlow backend
        z = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

        # Branch off a DNN to classify the latent vectors. Minimize cross-entropy loss
        if isCEloss:
            b = Concatenate()([z_mean, z_log_var])
            b = Dense(100, activation='relu')(b)
            b = Dense(100, activation='relu')(b)
            b = Dense(n_spks, activation='softmax')(b)

        # We instantiate these layers separately so as to reuse them later
        # It is important to use 'linear' as activation as vae_ctr_loss() in GaussCtrlossLayer
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

        # If minimizing center loss, use the speaker centers in the latent space as input
        # Use GaussCtrlossLayer.vae_ctr_loss() or GaussLayer.vae_loss as the loss function
        if isCenterloss:
            center = Input(shape=(2*latent_dim,))
            y = GaussCtrlossLayer()([x, x_decoded_mean, x_decoded_lvar, z_log_var, z_mean, center])
        else:
            y = GaussLayer()([x, x_decoded_mean, x_decoded_lvar, z_log_var, z_mean])

        # Create the VAE model with KL loss, Gaussian loss, and optionally center loss and cross-en loss
        # Passing loss=None to compile() causes the custom loss (vae_ctr_loss) to be used
        if isCenterloss and isCEloss:
            vae = Model(inputs=[x, center], outputs=[y, b])        
            vae.compile(optimizer='adam', loss='categorical_crossentropy', loss_weights=[1, 1])
        elif isCenterloss and not isCEloss:
            vae = Model(inputs=[x, center], outputs=y)        
            vae.compile(optimizer='adam', loss=None)
        elif not isCenterloss and isCEloss:
            vae = Model(inputs=x, outputs=[y, b])        
            vae.compile(optimizer='adam', loss='categorical_crossentropy', loss_weights=[1, 1])
        else:
            vae = Model(inputs=x, outputs=y)        
            vae.compile(optimizer='adam', loss=None)

        # Build a model to project inputs on the latent space (encoder)
        encoder = Model(inputs=x, outputs=Concatenate()([z_mean, z_log_var]))
        #encoder = Model(inputs=x, outputs=z_mean)

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


    # Train the VAE using x_train and validate it using x_test
    def fit(self, x_train, x_test, trn_ctr=None, tst_ctr=None, trn_lbs_1h=None, tst_lbs_1h=None, 
            n_epochs=10, batch_size=100, isCenterloss=False, isCEloss=False):
        self.check_if_model_exist()

        if isCenterloss and isCEloss:
            self.vae.fit(x=[x_train, trn_ctr], y=[x_train, trn_lbs_1h], shuffle=True, epochs=n_epochs, batch_size=batch_size,
                    validation_data=([x_test, tst_ctr], [x_test, tst_lbs_1h]))
        elif isCenterloss and not isCEloss:
            self.vae.fit(x=[x_train, trn_ctr], y=x_train, shuffle=True, epochs=n_epochs, batch_size=batch_size,
                    validation_data=([x_test, tst_ctr], x_test))
        elif not isCenterloss and isCEloss:
            self.vae.fit(x=x_train, y=[x_train, trn_lbs_1h], shuffle=True, epochs=n_epochs, batch_size=batch_size,
                    validation_data=(x_test, [x_test, tst_lbs_1h]))
        else:
            self.vae.fit(x=x_train, y=x_train, shuffle=True, epochs=n_epochs, batch_size=batch_size,
                    validation_data=(x_test, x_test))
        return self


    # Train the VAE using x_train and validate it using x_test
    def fit1(self, x_train, x_test, trn_ctr=None, tst_ctr=None, trn_lbs_1h=None, tst_lbs_1h=None, 
            n_epochs=10, batch_size=100, isCenterloss=False, isCEloss=False, x_train_ctr=None):
        self.check_if_model_exist()

        if isCenterloss and isCEloss:
            self.vae.fit(x=[x_train, trn_ctr], y=[x_train_ctr, trn_lbs_1h], shuffle=True, epochs=n_epochs, batch_size=batch_size,
                    validation_data=([x_test, tst_ctr], [x_test, tst_lbs_1h]))
        elif isCenterloss and not isCEloss:
            self.vae.fit(x=[x_train, trn_ctr], y=x_train, shuffle=True, epochs=n_epochs, batch_size=batch_size,
                    validation_data=([x_test, tst_ctr], x_test))
        elif not isCenterloss and isCEloss:
            self.vae.fit(x=x_train, y=[x_train, trn_lbs_1h], shuffle=True, epochs=n_epochs, batch_size=batch_size,
                    validation_data=(x_test, [x_test, tst_lbs_1h]))
        else:
            self.vae.fit(x=x_train, y=x_train, shuffle=True, epochs=n_epochs, batch_size=batch_size,
                    validation_data=(x_test, x_test))
        return self



    # Define our own mini-batch for training
    # Note ready. Need further development
    def train_on_batch(self, x_trn, x_tst, trn_ctr=None, tst_ctr=None, n_epochs=10, batch_size=100, isCenterloss=False):
        self.check_if_model_exist()
        n_batches = int(x_trn.shape[0] / batch_size)
        for epoch in range(n_epochs):
            loss = 0
            pidx = np.random.permutation(x_trn.shape[0])  # Random selection of vecs to create mini-batches
            if isCenterloss:
                for batch in range(n_batches):
                    idx = pidx[batch*batch_size: (batch+1)*batch_size]
                    self.vae.train_on_batch(x=[x_trn[idx,:], trn_ctr[idx]], y=None)
                    loss = loss + self.vae.test_on_batch(x=[x_tst[idx,:], tst_ctr[idx]], y=None)                
            else:    
                for batch in range(n_batches):
                    idx = pidx[batch*batch_size: (batch+1)*batch_size]
                    self.vae.train_on_batch(x=x_trn[idx,:], y=None)
                    loss = loss + self.vae.test_on_batch(x=x_tst[idx], y=None)
            print('Epoch=%d; Loss=%.3f' % (epoch, loss/n_batches))
        return self    

   # Train the VAE using speaker-dependent batch. Performance not good.
    def train_on_speaker(self, x_trn, labels, n_epochs=10):
        self.check_if_model_exist()
        n_spks = np.max(labels) + 1
        for epoch in range(n_epochs):
            loss = 0
            for s in range(n_spks):
                idx = [i for i, e in enumerate(labels) if e == s]
                loss = loss + self.vae.train_on_batch(x=x_trn[idx,:], y=None)
            print('Epoch=%d; Loss=%.3f' % (epoch, loss/n_spks))
        return self       

    # Perform end-to-end transform of X using the VAE
    def transform(self, X, batch_size=100):
        return self.vae.predict(X, batch_size=batch_size)

    # Encode input X using the encoder
    def encode(self, X, batch_size=100):
        self.encoder.predict(X, batch_size=batch_size)    


# A class implementing the custom loss layer for Gaussian input,
# The call() method defines the operation.
# See https://keras.io/layers/writing-your-own-keras-layers/
class GaussLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GaussLayer, self).__init__(**kwargs)

    def vae_loss(self, x, x_dec_mean, x_dec_lvar, zlogvar, zmean):
        diag_stdev = tf.exp(x_dec_lvar/2)
        dist = MultivariateNormalDiag(x_dec_mean, diag_stdev)
        gauss_loss = - dist.log_prob(x)       # Eq. 12 of Kingma (2014), L=1
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


# A class implementing the custom loss layer (with center loss) for Gaussian input,
class GaussCtrlossLayer(Layer):
    def __init__(self, **kwargs):
        self.is_placeholder = True
        super(GaussCtrlossLayer, self).__init__(**kwargs)

    def vae_ctr_loss(self, x, x_dec_mean, x_dec_lvar, zlogvar, zmean, center):
        diag_stdev = tf.exp(x_dec_lvar/2)
        dist = MultivariateNormalDiag(x_dec_mean, diag_stdev)
        gauss_loss = - dist.log_prob(x)       # Eq. 12 of Kingma (2014), L=1
        kl_loss = - 0.5 * K.sum(1 + zlogvar - K.square(zmean) - K.exp(zlogvar), axis=-1)
        z = Concatenate()([zmean, zlogvar])
        ctr_loss = K.sum(K.square(z - center), axis=-1)
        return K.mean(gauss_loss + kl_loss + 5*ctr_loss)   

    # Define the operation of the custom layer in the call() method
    def call(self, inputs):
        x = inputs[0]
        x_dec_mean = inputs[1]
        x_dec_lvar = inputs[2]        
        zlogvar = inputs[3]
        zmean = inputs[4]
        center = inputs[5]
        loss = self.vae_ctr_loss(x, x_dec_mean, x_dec_lvar, zlogvar, zmean, center)
        self.add_loss(loss, inputs=inputs)      # add_loss() is a function in keras.layers.Layer
        return x


# Main function for testing the GaussianVAE class
def main():
    GaussianVAE(input_dim=512, hidden_dim=300, latent_dim=200, epsilon_std=1)


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