# A python class implementing a Variational Autoencoder (VAE) that creates
# a latent space for speaker identification. The input can be either i-vectors
# or x-vectors.
# The vectors in the embedding layer are used as features for determining
# speaker IDs of test x-vectors using cosine-distances
# Speaker IDs can be extracted from Kaldi's spk2utt under the 'data/dataset' folder.
# For example, 'data/swbd/spk2utt' contains <spkid> <xvec-id1> <xvec-id2>...,
# where <xvec-id> match the key field of exp/xvectors_swbd/xvector.scp.
#
# To run this script using Python3.6 (enmcomp3,4,11,13),
# assuming that Anaconda3 environment "tf-py3.6"
# has been created already
#   bash
#   export PATH=/usr/local/anaconda3/bin:/usr/local/cuda-8.0/bin:$PATH
#   export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-7.5/lib64
#   source activate tf-py3.6
#   python3 xvec_vae_emb.py
#   source deactivate tf-py3.6

# M.W. Mak, Nov. 2018

# Performance of speaker identification on the test domain (dom_tst)
# dom_trn = ['swbd2', 'swbdcell', 'sre04-10-mx6', 'sre16', 'sitw', 'voxceleb1', 'sre16_eval']
# dom_tst = ['sre18_dev_cmn2']
# No. of hidden nodes = 800
# No. of epochs = 200
# Acc using raw x-vectors = 81.85%
#                            Latent Dim
# Transformation Method   Ctr-loss  CE-loss    64          128         256
# --------------------------------------------------------------------------
# None                                         81.85       81.85       81.85
# PCA                     N/A       N/A        79.04       82.19       82.94
# LDA                     N/A       N/A        95.29       96.78       97.36
#
# VAE                       Y         Y        90.98       93.62       92.76
# VAE                       N         N                    93.11
# VAE                       N         Y                    93.11
# VAE                       Y         F                    93.80
#
# VAE + LDA                 Y         Y        95.52       97.24       96.90
# VAE + LDA                 N         N                    96.96
# VAE + LDA                 N         Y                    97.53
# VAE + LDA                 Y         N                    97.42 

from __future__ import print_function

import tensorflow as tf
from myPlot import scatter2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from keras.utils import np_utils
from util_func import load_xvectors, get_accuracy, select_speakers, save_xvectors_h5, load_xvectors_h5
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.decomposition import PCA
from GaussianVAE import GaussianVAE
import os

# Main function
def main():
    # Define some constants
    isDebug = True
    isCenterloss = True         # Minimize center loss in the latent space
    isCEloss = True             # Minimize cross-entropy loss in the latent space
    create_tsne = False         # Create t-SNE plot or not
    hidden_dim = 800            # Dimension of hidden layer in the VAE
    latent_dim = 128            # Dimension of the latent (encoded/embedded) layer of the VAE
    e_std = 0.01                 # Standard deviation of Gaussian in the sampling layer
    expdir = '../v2-1/exp/'     # Directory storing x-vectors from different domains
    vectype = 'xvector'         # Type of vectors, either 'ivector' or 'xvector'
    datadir = '../v2-1/data/'   # Kaldi data dirctory storing info (e.g. spk2utt) of individual domains
    basedir = '/home5a/mwmak/so/spkver/sre18-dev/v2-1/'   # Base dir of Kaldi evaluation
    min_n_trn_vecs = 30             # Min number of x-vectors per speaker for training adversarial network
    min_n_tst_vecs = 30            # Min number of x-vectors per speaker for testing adversarial network
    n_sel_spks = 10              # No. of selected speakers per domain for t-sne plots
    if isDebug:    
        dom_trn = ['swbd', 'sre16_eval']
        dom_tst = ['sre18_dev_cmn2']
        n_epochs = 10
        trn_file_h5 = 'data/h5/xvectors_2domain.h5'
        tst_file_h5 = 'data/h5/xvectors_1domain.h5'
    else:
        dom_trn = ['swbd2', 'swbdcell', 'sre04-10-mx6', 'sre16', 'sitw', 'voxceleb1', 'sre16_eval']
        dom_tst = ['sre18_dev_cmn2']
        n_epochs = 200
        trn_file_h5 = 'data/h5/xvectors_7domain.h5'
        tst_file_h5 = 'data/h5/xvectors_1domain.h5'

    # Load x-vectors from expdir/domains and in-domain
    if os.path.isfile(trn_file_h5):
        x_trn, spk_lbs_trn, dom_lbs_trn, _ = load_xvectors_h5(trn_file_h5)
    else:    
        x_trn, spk_lbs_trn, dom_lbs_trn, utt_ids_trn = load_xvectors(dom_trn, expdir, vectype, datadir, basedir, min_n_vecs=min_n_trn_vecs)
        save_xvectors_h5(trn_file_h5, x_trn, spk_lbs_trn, dom_lbs_trn, utt_ids_trn)
    if os.path.isfile(tst_file_h5):
        x_tst, spk_lbs_tst, dom_lbs_tst, utt_ids_tst = load_xvectors_h5(tst_file_h5)
    else:    
        x_tst, spk_lbs_tst, dom_lbs_tst, utt_ids_tst = load_xvectors(dom_tst, expdir, vectype, datadir, basedir, min_n_vecs=min_n_tst_vecs)
        save_xvectors_h5(tst_file_h5, x_tst, spk_lbs_tst, dom_lbs_tst, utt_ids_tst)

    # Display train and test info    
    n_trn_spk = int(np.max(spk_lbs_trn)+1)
    n_trn_dom = int(np.max(dom_lbs_trn)+1)
    print('No. of training x-vectors = %d' % int(x_trn.shape[0]))
    print('No. of training speakers = %d' % n_trn_spk)
    print('No. of training domains = %d' % n_trn_dom)
    print('No. of test x-vectors = %d' % int(x_tst.shape[0]))
    print('No. of test speakers = %d' % int(np.max(spk_lbs_tst)+1))
    print('No. of test domains = %d' % int(np.max(dom_lbs_tst)+1))

    # Convert to one-hot encoding
    spk_lbs_trn_1h = np_utils.to_categorical(spk_lbs_trn)
    spk_lbs_tst_1h = np_utils.to_categorical(spk_lbs_tst)

    # Select speakers from test domains for producing t-sne plots
    x_sel, spk_lbs_sel, dom_lbs_sel, _ = select_speakers(x_tst, spk_lbs_tst, dom_lbs_tst, utt_ids_tst, min_n_vecs=20, n_spks=n_sel_spks)

    # Create and train a VAE. If center loss is true, train VAE twice; 
    # the first time (without ctr loss) is to find the speaker centers in the latent space
    if isCenterloss:
        gvae = GaussianVAE(input_dim=x_trn.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim, 
                            epsilon_std=e_std, isCenterloss=False, isCEloss=isCEloss, n_spks=n_trn_spk)
        gvae.fit(x_trn, x_trn, trn_lbs_1h=spk_lbs_trn_1h, tst_lbs_1h=spk_lbs_trn_1h, 
                    n_epochs=n_epochs, batch_size=100, isCenterloss=False, isCEloss=isCEloss)
        gvae.vae.save('models/vae.hd5')
        z_trn = gvae.encoder.predict(x_trn)
        z_trn_ctr = get_spk_centers(z_trn, spk_lbs_trn)
        gvae = GaussianVAE(input_dim=x_trn.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim, 
                            epsilon_std=e_std, isCenterloss=True, isCEloss=isCEloss, n_spks=n_trn_spk)
        gvae.vae.load_weights('models/vae.hd5')                    
        gvae.fit(x_trn, x_trn, z_trn_ctr, z_trn_ctr, spk_lbs_trn_1h, spk_lbs_trn_1h, 
                    n_epochs=n_epochs, batch_size=100, isCenterloss=True, isCEloss=isCEloss)
    else:
        gvae = GaussianVAE(input_dim=x_trn.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim, 
                            epsilon_std=e_std, isCenterloss=False, isCEloss=isCEloss, n_spks=n_trn_spk)
        gvae.fit(x_trn, x_trn, trn_ctr=None, tst_ctr=None, trn_lbs_1h=spk_lbs_trn_1h, tst_lbs_1h=spk_lbs_trn_1h,
                    n_epochs=n_epochs, batch_size=100, isCenterloss=False, isCEloss=isCEloss)

    # Get the encoder
    encoder = gvae.encoder

    # Use VAE to transform the test x-vectors to the embedding space
    x_tst_vae = encoder.predict(x_tst)

    # VAE + LDA
    x_trn_vae = encoder.predict(x_trn)
    x_tst_vaelda = LinearDiscriminantAnalysis(n_components=latent_dim).fit(x_trn_vae, spk_lbs_trn).transform(x_tst_vae)
    x_tst_vaelda = np.hstack((x_tst_vae, x_tst_vaelda))

    # Use PCA to transform the x-vectors
    x_tst_pca =  PCA(n_components=latent_dim).fit(x_trn).transform(x_tst)

    # Use LDA to transform the x-vectors
    x_tst_lda =  LinearDiscriminantAnalysis(n_components=latent_dim).fit(x_trn, spk_lbs_trn).transform(x_tst)

    # Free up memory
    del x_trn

    # Use cosine distance to determine the speaker IDs of VAE-transformed x-vectors
    y_spk_vae = cosine_scoring(x_tst_vae, spk_lbs_tst)
    print('Speaker identificiation acc using VAE x-vecs = %.2f%%' % get_accuracy(y_spk_vae, spk_lbs_tst_1h))

    # Use cosine distance to determine the speaker IDs of VAE+LDA-transformed x-vectors
    y_spk_vaelda = cosine_scoring(x_tst_vaelda, spk_lbs_tst)
    print('Speaker identificiation acc using VAE+LDA x-vecs = %.2f%%' % get_accuracy(y_spk_vaelda, spk_lbs_tst_1h))

    # Use cosine distance to determine the speaker IDs of PCA-transformed x-vectors
    y_spk_pca = cosine_scoring(x_tst_pca, spk_lbs_tst)
    print('Speaker identificiation acc using PCA x-vecs = %.2f%%' % get_accuracy(y_spk_pca, spk_lbs_tst_1h))

    # Use cosine distance to determine the speaker IDs of LDA-transformed x-vectors
    y_spk_lda = cosine_scoring(x_tst_lda, spk_lbs_tst)
    print('Speaker identificiation acc using LDA x-vecs = %.2f%%' % get_accuracy(y_spk_lda, spk_lbs_tst_1h))

    # Use cosine distance to determine the speaker IDs of original x-vectors
    y_spk = cosine_scoring(x_tst, spk_lbs_tst)
    print('Speaker identificiation acc using original x-vecs = %.2f%%' % get_accuracy(y_spk, spk_lbs_tst_1h))

    # Use cosine distance to determine the domain of VAE x-vectors
    #y_dom_vae = cosine_scoring(x_tst_vae, dom_lbs_tst)
    #print('Domain classification acc using VAE x-vecs = %.2f%%' % get_accuracy(y_dom_vae, dom_lbs_tst_1h))

    # Use cosine distance to determine the domain of PCA x-vectors
    #y_dom_pca = cosine_scoring(x_tst_pca, dom_lbs_tst)
    #print('Domain classification acc using PCA x-vecs = %.2f%%' % get_accuracy(y_dom_pca, dom_lbs_tst_1h))

    # Use cosine distance to determine the domain of original x-vectors
    #y_dom = cosine_scoring(x_tst, dom_lbs_tst)
    #print('Domain classification acc using PCA x-vecs = %.2f%%' % get_accuracy(y_dom, dom_lbs_tst_1h))

    # Save models to .h5 files
    encoder.save('models/vae_emb-%d-%d-epoch%d.h5' % (latent_dim, min_n_trn_vecs, n_epochs))

    # Plot original and VAE-projected x-vectors on 2-D t-SNE space 
    if create_tsne == True:
        print('Creating t-SNE plot of original x-vectors')
        x_prj = TSNE(random_state=20150101).fit_transform(x_sel)
        fig, _, _, _ = scatter2D(x_prj, spk_lbs_sel, markers=dom_lbs_sel, n_colors=np.max(spk_lbs_sel)+1,
                                title='Original test x-vectors')
        fig.savefig('logs/xvec_tst.png')
        plt.show(block=False)

        print('Creating t-SNE plot of VAE-projected x-vectors')
        x_sel_enc = TSNE(random_state=20150101).fit_transform(encoder.predict(x_sel))
        fig, _, _, _ = scatter2D(x_sel_enc, spk_lbs_sel, markers=dom_lbs_sel, n_colors=np.max(spk_lbs_sel)+1,
                            title='VAE Transformed X-Vectors (Epoch = %d)' % n_epochs)
        filename = 'logs/vae_emb-%d-%d-epoch%d.png' % (latent_dim, min_n_trn_vecs, n_epochs)
        fig.savefig(filename)
        plt.show(block=True)


def get_spk_centers(x, lbs):
    centers = np.zeros(shape=x.shape)
    n_spks = np.max(lbs) + 1
    for k in range(n_spks):
        idx = [j for j, e in enumerate(lbs) if e == k]
        ctr = np.mean(x[idx,:], axis=0)
        centers[idx,:] = np.tile(ctr, (len(idx), 1))            # Repeat ctr a number of times
    return centers

def cosine_scoring(x, lbs):
    return cosine_scoring2(x, lbs)


# For each test vector, score it against the mean vectors of each speaker. Return a score
# matrix of size (n_vecs, n_spks)
def cosine_scoring2(x, lbs):
    n_vecs = x.shape[0]
    n_spks = np.max(lbs) + 1
    scrmat = np.zeros((n_vecs, n_spks))
    for i in range(n_vecs):
        tgt_x = list()
        for k in range(n_spks):
            idx = [j for j, e in enumerate(lbs) if e == k and j != i]
            tgt_x.append(np.mean(x[idx,:], axis=0))
        tgt_x = np.vstack(tgt_x)
        for k in range(n_spks):
            scrmat[i,k] = np.dot(x[i,:],tgt_x[k,:])/(np.linalg.norm(x[i,:])*np.linalg.norm(tgt_x[k,:]))
    return scrmat


def cos_dist(x, y):
    return np.dot(x, y)/(np.linalg.norm(x)*np.linalg.norm(y))


# For each test vector, score it against all other test vectors. Then, for each speaker, find the
# maximum score for this test vector and assign the maximum score to the score matrix of size
# (n_vecs, n_spks)
def cosine_scoring1(x, lbs):
    n_vecs = x.shape[0]
    scr_mat = np.zeros((n_vecs, n_vecs))
    for i in range(n_vecs):
        for j in range(i+1,n_vecs):
            scr_mat[i,j] = np.dot(x[i,:],x[j,:])/(np.linalg.norm(x[i,:])*np.linalg.norm(x[j,:]))
            scr_mat[j,i] = scr_mat[i,j]
    y_pred = list()        
    for i in range(n_vecs):
        scr = list()
        for k in np.unique(lbs):
            idx = [j for j, e in enumerate(lbs) if e == k]
            scr.append(np.max(scr_mat[i,idx]))
        y_pred.append(scr)
    return np.array(y_pred) 


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


