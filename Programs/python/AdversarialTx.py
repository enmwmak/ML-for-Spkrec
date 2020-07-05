# Train an adversarial transformer
# Data can be MNIST digits or others, such as i-vectors
# For MNIST data, produce images to the folder ./images/aae
# Derived from https://github.com/eriklindernoren/Keras-GAN/blob/master/aae/adversarial_autoencoder.py

from __future__ import print_function
from keras.layers import Input, Dense, Dropout
from keras.layers import BatchNormalization, concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import numpy as np
from keras import backend as K
from keras.utils import plot_model

# Class of adversarial transformer (Fig. 2 of 2017/18 GRF proposal)
class AdversarialTx:
    def __init__(self, savefunc=None, logdir='logs', f_dim=300, z_dim=100, n_cls=10, n_dom=2, 
                    enc_hnode=[512,512], cls_hnode=[512,512], dis_hnode=[64]):
        self.f_dim = f_dim           # Dim of i-vectors.
        self.logdir = logdir         # Directory for storing log files (if any)
        self.savefunc = savefunc     # User defined function to save the generated data
        self.encoded_dim = z_dim     # Dim of encoded representation z
        self.n_classes = n_cls       # No. of classes (speakers) in the i-vectors
        self.n_domains = n_dom       # No. of domains in the i-vectors
        self.enc_hnode = enc_hnode   # No. of hidden nodes in each hidden layer of encoder
        self.cls_hnode = cls_hnode   # No. of hidden nodes in each hidden layer of classifier
        self.dis_hnode = dis_hnode   # No. of hidden nodes in each hidden layer of discriminator
        self.encoder = None
        self.discriminator = None
        self.classifier = None
        self.generator = None

        # Create an optimizer
        optimizer = Adam(0.0002, 0.5)

        # Build the discriminator
        self.discriminator = self.build_discriminator()
        self.discriminator.summary()

        # Build the generator, which comprises a feature encoder and a speaker classifier.
        # The generator takes i-vec as input and produces encoded representation with
        # dimension encoded_dim and spk posteriors with dim equals to n_classes
        self.generator = self.build_generator()
        self.generator.summary()
        x = Input(shape=(self.f_dim,))
        x_encoded, y_spk = self.generator(x)

        # Build the adversarial transformer, which comprises the feature encoder G(x),
        # the speaker classifier C(G(x), and the discriminator D(G(x))
        # When training the transformer, we will only unpdate G and C but keep D fixed.
        # We have two options for the loss function
        # (1) We pass a single loss to compile so that all output use the same loss function.
        #     This requires us to make the effect of gradient reversal during the call to train_on_batch().
        #     This may be achieved by flipping the target outputs if the discriminator has two output only.
        #     But the objective function is not exact as -\sum_i(1-t_i)log y_i is not equal to
        #     \sum_i t_i * log y_i
        # (2) A better way is to define our own loss function, we have cross-entropy as usual for
        #     classifier's loss but negative of the cross-entropy for discriminator loss. But the loss and
        #     accuracy returned from train_on_batch are different from option 1
        self.discriminator.trainable = False
        y_dom = self.discriminator(x_encoded)
        self.adv_transformer = Model(inputs=x, outputs=[y_spk, y_dom])
        # self.adv_transformer.compile(loss='categorical_crossentropy', loss_weights=[0.5, 0.5],
        #                              optimizer=optimizer, metrics=['categorical_accuracy'])
        self.adv_transformer.compile(loss=[pos_cat_cxe, neg_cat_cxe], loss_weights=[1, 0.1],
                                     optimizer=optimizer, metrics=['accuracy'])

        # Compile the discriminator. 
        # As metrics is 'accuracy', train_on_batch() will return cat_cxe and acc as loss
        self.discriminator.trainable = True
        self.discriminator.compile(loss='categorical_crossentropy', optimizer=optimizer,
                                   metrics=['accuracy'])

        plot_model(self.adv_transformer, to_file='logs/adv_transformer.png', show_shapes=True, 
                    show_layer_names=True)


    # Generator is a DNN with arch: 300-128-128-(100)-128-128-10 (for 10-class problem)
    # It comprises a feature encoder G(x) and a speaker classifier C(G(x)
    def build_generator(self):
        # Feature encoder
        encoder = Sequential(name='Enc-seq-model')
        for n in self.enc_hnode:
            encoder.add(Dense(n, input_dim=self.f_dim, kernel_initializer='glorot_normal'))
            encoder.add(LeakyReLU(alpha=0.2))
            encoder.add(BatchNormalization(momentum=0.8))
        encoder.add(Dense(self.encoded_dim, name='Embedded-layer', kernel_initializer='glorot_normal',
                          activation='linear'))
        encoder.add(BatchNormalization(momentum=0.8))
        x = Input(shape=(self.f_dim,), name='Enc-in-x')
        x_encoded = encoder(x)
        self.encoder = Model(inputs=x, outputs=x_encoded, name='Encoder')

        # Speaker Classifier:
        z = Input(shape=(self.encoded_dim,), name='Cls-in-z')
        y = z
        for n in self.cls_hnode:
            y = Dense(n, kernel_initializer='glorot_normal')(y)
            y = Dropout(0.2)(y)
            y = LeakyReLU(alpha=0.2)(y)
            y = BatchNormalization(momentum=0.8)(y)
        y = Dense(self.n_classes, activation='softmax', name='Cls-out', kernel_initializer='glorot_normal')(y)
        self.classifier = Model(inputs=z, outputs=y, name='Classifier')

        # Connecting the encoder and classifier
        y_spk = self.classifier(x_encoded)

        # Return a composite network (Generator) that outputs x_encoded and y_spk
        generator = Model(inputs=x, outputs=[x_encoded, y_spk], name='Generator')
        return generator


    # Discriminator is a DNN with arch: 100-128-128-2 (for 2 domains)
    def build_discriminator(self):
        model = Sequential(name='Disc-seq-model')
        for n in self.dis_hnode:
            model.add(Dense(n, input_dim=self.encoded_dim, kernel_initializer='glorot_normal'))
            model.add(Dropout(0.2))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.n_domains, activation='softmax', name='Disc-out', kernel_initializer='glorot_normal'))
        x_encoded = Input(shape=(self.encoded_dim,), name='Disc-in-z')
        y_dom = model(x_encoded)

        # Return a DNN with x_encoded as input and y_dom as output
        discriminator = Model(inputs=x_encoded, outputs=y_dom, name='Discriminator')
        return discriminator


    # Train the adversarial network
    def train(self, x_trn, y_spk_trn_1h, y_dom_trn_1h, n_epochs=20, batch_size=128):
        n_batches = int(x_trn.shape[0] / batch_size)
        for epoch in range(n_epochs):
            d_loss_cxe = 0              # Accumulated cross-entropy loss of discriminator
            d_loss_acc = 0              # Accumulated loss of discriminator for computing accuracy
            t_loss_cxe1 = 0             # Accumulated cross-entropy loss in the speaker classifier
            t_loss_cxe2 = 0
            pidx = np.random.permutation(x_trn.shape[0])  # Random selection of vecs to create mini-batches
            for batch in range(n_batches):

                # Select a batch of vectors and class labels in 1-hot format
                idx = pidx[batch*batch_size: (batch+1)*batch_size]
                x = x_trn[idx]
                y_dom_1h = y_dom_trn_1h[idx]
                y_spk_1h = y_spk_trn_1h[idx]

                # ---------------------------------------------------------------
                #  Train Discriminator using x as input and y_dom_1h as target
                # ---------------------------------------------------------------
                x_encoded, _ = self.generator.predict(x)
                d_loss = self.discriminator.train_on_batch(x_encoded, y_dom_1h)
                d_loss_cxe = d_loss_cxe + d_loss[0]
                d_loss_acc = d_loss_acc + d_loss[1]

                # ------------------------------------------------------------------------------
                #  Train the generator part (encoder+classifier) of the adversarial transformer
                # ------------------------------------------------------------------------------
                # The feature encoder wants to confuse the discriminator in classifying the domain
                # This is achieved by performing gradient ascend on the discriminator's cross-entropy error.
                # y_flipsex_1h = np.flip(y_dom_1h, axis=1)
                for i in range(3):
                    t_loss = self.adv_transformer.train_on_batch(x, [y_spk_1h, y_dom_1h])
                t_loss_cxe1 = t_loss_cxe1 + t_loss[0]
                t_loss_cxe2 = t_loss_cxe2 + t_loss[1]

            # Show training progress
            d_loss_cxe = d_loss_cxe/n_batches
            d_loss_acc = d_loss_acc/n_batches * 100
            t_loss_cxe1 = t_loss_cxe1/n_batches
            t_loss_cxe2 = t_loss_cxe2/n_batches
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f, %f]" %
                  (epoch, d_loss_cxe, d_loss_acc, t_loss_cxe1, t_loss_cxe2))

            y_spk_pred, y_dom_pred = self.generator.predict(x)
            y_pred = np.hstack([y_spk_pred, y_dom_pred])
            filename = self.logdir + '/' + "%d.log" % epoch
            if self.savefunc is not None:
                self.savefunc(filename, y_pred)


    def get_generator(self):
        return self.generator

    def generate(self, x, y):
        return self.generator.predict([x, y])


    # Return the trained encoder. Can be called after training.
    def get_encoder(self):
        return self.encoder

    # Return the trained decoder. Can be called after training
    def get_decoder(self):
        decoder = Model(inputs=self.generator.get_layer('Decoder').get_input_at(0),
                        outputs=self.generator.get_layer('Decoder').get_output_at(0))
        return decoder

    # Return the classifier
    def get_classifier(self):
        return self.classifier

def pos_cat_cxe(y_true, y_pred):
    return K.categorical_crossentropy(y_true, y_pred)


def neg_cat_cxe(y_true, y_pred):
    return -1 * K.categorical_crossentropy(y_true, y_pred)
