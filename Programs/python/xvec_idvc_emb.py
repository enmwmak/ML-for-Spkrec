# A python class implementing IDVC to transform x-vector to domain-invariant space.
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
#   python3 xvec_idvc_emb.py
#   source deactivate tf-py3.6

# M.W. Mak, Oct. 2018

from __future__ import print_function

import tensorflow as tf
from myPlot import scatter2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from keras.utils import np_utils
from util_func import load_xvectors, get_accuracy, select_speakers, save_xvectors_h5, load_xvectors_h5
from kaldi_arkwrite import save_as_ark_scp
import os
from idvc import IDVC

# Main function
def main():
    
    print("Make sure that exp/ is linked to exp_original/")
    os.remove('../v2-1/exp')
    os.symlink('../v2-1/exp_original', '../v2-1/exp')
    
    # Define some constants
    expdir = '/home5a/mwmak/so/spkver/sre18-dev/v2-1/exp_original/'     # Directory storing x-vectors from different domains
    vectype = 'xvector'         # Type of vectors, either 'ivector' or 'xvector'
    datadir = '/home5a/mwmak/so/spkver/sre18-dev/v2-1/data/'   # Kaldi data dirctory storing info (e.g. spk2utt) of individual domains
    basedir = '/home5a/mwmak/so/spkver/sre18-dev/v2-1/'   # Base dir of Kaldi evaluation
    idvcfile = 'models/idvc.pkl'                        # Storing IDVC object
    min_n_vecs = 10             # Min number of x-vectors per speaker for training IDVC transformation matrix
    n_sel_spks = 5             # No. of speakers for t-SNE plot
    
    dom_trn = ['swbd2', 'swbdcell', 'sre04-10-mx6', 'sre16', 'sitw_eval', 'voxceleb1'] 
    #dom_trn = ['sre16_eval', 'sitw_eval']        # For debugging
    dom_tst = ['sre18_dev', 'sitw_dev']
    trn_file_h5 = 'data/h5/xvectors_idvc_train.h5'      # 5 domains, at least 10 utts/spk, see scp2h5.py
    tst_file_h5 = 'data/h5/xvectors_idvc_test.h5'       # 2 domains, at least 10 utts per speaker    
    whiten_method = 'WCCN'                              # Whitening method applied to IDVC-transformed vectors ('PCA'|'WCCN')
    
    # Load x-vectors from expdir/domains and in-domain
    if os.path.isfile(trn_file_h5):
        x_trn, spk_lbs_trn, dom_lbs_trn, _ = load_xvectors_h5(trn_file_h5)
    else:    
        x_trn, spk_lbs_trn, dom_lbs_trn, utt_ids_trn = load_xvectors(dom_trn, expdir, vectype, datadir, basedir, min_n_vecs=min_n_vecs)
        save_xvectors_h5(trn_file_h5, x_trn, spk_lbs_trn, dom_lbs_trn, utt_ids_trn)
    if os.path.isfile(tst_file_h5):
        x_tst, spk_lbs_tst, dom_lbs_tst, utt_ids_tst = load_xvectors_h5(tst_file_h5)
    else:    
        x_tst, spk_lbs_tst, dom_lbs_tst, utt_ids_tst = load_xvectors(dom_tst, expdir, vectype, datadir, basedir, min_n_vecs=min_n_vecs)
        save_xvectors_h5(tst_file_h5, x_tst, spk_lbs_tst, dom_lbs_tst, utt_ids_tst)
    n_trn_spk = int(np.max(spk_lbs_trn)+1)
    n_trn_dom = int(np.max(dom_lbs_trn)+1)

    print('No. of training x-vectors = %d' % int(x_trn.shape[0]))
    print('No. of training speakers = %d' % n_trn_spk)
    print('No. of training domains = %d' % n_trn_dom)
    print('No. of test x-vectors = %d' % int(x_tst.shape[0]))
    print('No. of test speakers = %d' % int(np.max(spk_lbs_tst)+1))
    print('No. of test domains = %d' % int(np.max(dom_lbs_tst)+1))

    # Convert to one-hot encoding
    spk_lbs_tst_1h = np_utils.to_categorical(spk_lbs_tst)
    dom_lbs_tst_1h = np_utils.to_categorical(dom_lbs_tst)

    # Select speakers from test domains for producing t-sne plots
    x_sel, spk_lbs_sel, dom_lbs_sel, utt_ids_sel = select_speakers(x_tst, spk_lbs_tst, 
                                dom_lbs_tst, utt_ids_tst, min_n_vecs=20, n_spks=n_sel_spks)

    # Use cosine distance to determine the speaker IDs of IDVC-transformed test x-vectors
    print('Performing IDVC with %s whitening' % whiten_method)
    fdim = x_trn.shape[-1]
    idvc = IDVC(fdim, whiten=True, whiten_method=whiten_method).fit(x_trn, dom_lbs_trn, spk_lbs=spk_lbs_trn)    
    idvc.save_model(idvcfile)
    x_tst_enc = idvc.transform(x_tst)

    # Evaluate performance of selected test x-vectors    
    y_spk_pred = cosine_scoring(x_tst_enc, spk_lbs_tst)
    print('\nSpeaker identificiation acc using IDVC x-vecs = %.2f%%' % get_accuracy(y_spk_pred, spk_lbs_tst_1h))

    # Use cosine distance to determine the domain of IDVC-transformed test x-vectors
    y_dom_pred = cosine_scoring(x_tst_enc, dom_lbs_tst)
    print('Domain classification acc using IDVC x-vecs = %.2f%%' % get_accuracy(y_dom_pred, dom_lbs_tst_1h))

    # Use cosine distance to determine the speaker IDs of original test x-vectors
    y_spk_pred = cosine_scoring(x_tst, spk_lbs_tst)
    print('Speaker identificiation acc using original x-vecs = %.2f%%' % get_accuracy(y_spk_pred, spk_lbs_tst_1h))

    # Use cosine distance to determine the domains of original test x-vectors
    y_dom_pred = cosine_scoring(x_tst, dom_lbs_tst)
    print('Domain classification acc using original x-vecs = %.2f%%' % get_accuracy(y_dom_pred, dom_lbs_tst_1h))    

    # Save the IDVC-transformed test data as .ark and .scp file
    print('Save IDVC-transformed test vectors as .ark and .scp file')
    save_as_ark_scp(dict(zip(utt_ids_tst, x_tst_enc)), expdir + 'xvectors_test/xvector')

    # Plot original x-vectors on 2-D t-SNE space
    print('Creating t-SNE plot of x-vectors')
    x_prj = TSNE(random_state=20150101).fit_transform(x_sel)
    fig, _, _, _ = scatter2D(x_prj, spk_lbs_sel, markers=dom_lbs_sel, n_colors=np.max(spk_lbs_sel)+1,
                             title='Original x-vectors')
    fig.savefig('logs/xvec_tst.png')
    plt.show(block=False)

    # Plot IDVC-projected vectors of x_sel on t-SNE space
    x_sel_enc = idvc.transform(x_sel)
    x_sel_enc = TSNE(random_state=20150101).fit_transform(x_sel_enc)
    fig, _, _, _ = scatter2D(x_sel_enc, spk_lbs_sel, markers=dom_lbs_sel, n_colors=np.max(spk_lbs_sel)+1,
                             title='IDVC-Transformed X-Vectors')
    fig.savefig('logs/idvc_emb.png')
    plt.show(block=True)

    
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


