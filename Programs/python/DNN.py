# A python class for conventional DNN
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

class DNN:
    def __init__(self, savefunc=None, logdir='logs', f_dim=300, n_cls=10, hnode=[512,512]):
        self.f_dim = f_dim           # Dim of i-vectors.
        self.logdir = logdir         # Directory for storing log files (if any)
        self.savefunc = savefunc     # User defined function to save the generated data
        self.n_classes = n_cls       # No. of classes (speakers) in the i-vectors
        self.hnode = hnode           # No. of hidden nodes in each hidden layer 
        self.classifier = None

        # Create an optimizer
        optimizer = Adam(0.0002, 0.5)

        # Build the classifier
        self.classifier = self.build_classifier()
 
        # Compile the classifier
        self.classifier.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        self.classifier.summary()
        plot_model(self.classifier, to_file='logs/dnn.png', show_shapes=True, show_layer_names=True)


    # Classifier is a DNN with no. of nodes per hidden layer defined in self.hnode
    def build_classifier(self):
        model = Sequential()
        for n in self.hnode:
            model.add(Dense(n, input_dim=self.f_dim, kernel_initializer='glorot_normal'))
            model.add(LeakyReLU(alpha=0.2))
            model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(self.n_classes, activation='softmax', name='Cls-out', kernel_initializer='glorot_normal'))
        x = Input(shape=(self.f_dim,), name='Classifier-in')
        y = model(x)

        # Return a DNN with x as input and y as output
        self.classifier = Model(inputs=x, outputs=y, name='Classifier')
        return self.classifier


    # Train the DNN
    def train(self, x_trn, y_spk_trn_1h, n_epochs=20, batch_size=128):
        n_batches = int(x_trn.shape[0] / batch_size)
        for epoch in range(n_epochs):
            d_loss_cxe = 0              # Accumulated cross-entropy loss of the classifier
            d_loss_acc = 0              # Accumulated loss of classifier for computing accuracy
            pidx = np.random.permutation(x_trn.shape[0])  # Random selection of vecs to create mini-batches
            for batch in range(n_batches):

                # Select a batch of vectors and class labels in 1-hot format
                idx = pidx[batch*batch_size: (batch+1)*batch_size]
                x = x_trn[idx]
                y_spk_1h = y_spk_trn_1h[idx]

                # ---------------------------------------------------------------
                #  Train Classifier using x as input and y_spk_1h as target
                # ---------------------------------------------------------------
                d_loss = self.classifier.train_on_batch(x, y_spk_1h)
                d_loss_cxe = d_loss_cxe + d_loss[0]
                d_loss_acc = d_loss_acc + d_loss[1]

            # Show training progress
            d_loss_cxe = d_loss_cxe/n_batches
            d_loss_acc = d_loss_acc/n_batches * 100
            print("%d [Cross-entropy loss: %f, acc.: %.2f%%]" % (epoch, d_loss_cxe, d_loss_acc))

            y_spk_pred = self.classifier.predict(x)
            y_pred = np.hstack(y_spk_pred)
            filename = self.logdir + '/' + "%d.log" % epoch
            if self.savefunc is not None:
                self.savefunc(filename, y_pred)


    # Return the classifier
    def get_classifier(self):
        return self.classifier

