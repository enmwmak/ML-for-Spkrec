import platform
import sys
from mPLDA_train4_ import mPLDA_train4
import numpy as np
# import cupy as np     # use cuda to acclerate calculation if cuda devices are avilable
from wccn_ import wccn
from len_norm_ import len_norm
from ldawccn_ import ldawccn
from BUT2PLDA_ import BUT2PLDA
from scipy.io import loadmat, savemat
from utili_ import dict2h5

def remove_bad_ivec(X, L, spk_logical, normlimit):
    """
    Remove i-vecs with big norm
    """

    N = len(spk_logical)
    normX = np.zeros(shape=(N,1), dtype= np.float64)
    for i in range(L.shape[0]):
        normX[i]= np.linalg.norm(X[i, :])
    idx = np.where(normX < normlimit)[0]
    X = X[idx, :]
    spk_logical = spk_logical[idx]
    L = L[idx]
    return X, L, spk_logical

def remove_bad_spks(X, L, spk_logical, min_num_utts):
    """
     Remove speaker with less than 2 utts
    """

    _, _, spk_ids = np.unique(spk_logical, return_index=True, return_inverse=True)
    numSpks = len(np.unique(spk_ids))
    rm_idx = np.empty(shape=(0,), dtype=int)
    for i in range(numSpks):
        idx = np.where(spk_ids == i)[0]
        if len(idx) < min_num_utts:
            rm_idx = np.append(rm_idx, idx)
    spk_logical = np.delete(spk_logical, rm_idx)
    X = np.delete(X, rm_idx, axis=0)
    L = np.delete(L, rm_idx)
    return X, L, spk_logical

def comp_mGPLDA(X, L, spk_logical, GPLDA_file, n_ev, n_mix, mPLDA_trainfunc = mPLDA_train4):
    """
     function mGPLDAModel = comp_mGPLDA(X, L, spk_logical, GPLDA_file, n_ev, n_mix)
     Train an SNR-dependent mixture GPLDA model based on training i-vectors and speaker session info.
     This file implements the mPLDA model in my Interspeech14 paper.
     Use the GPLDA package in ~/so/Matlab/BayesPLDA
     Input:
       X            - training ivectors in rows
       L            - Either utterance length or SNR for each row in X
       spk_logical  - speaker session info (BUT JFA package)
       n_ev         - Dim of speaker space (No. of cols. in GPLDAModel.F)
       GPLDA_file   - .mat file storing the GPLDA model structure (output)
       n_mix        - No. of mixtures in GPLDA model
     Output:
       mGPLDAModel  - Structure containing mGPLDA model. It has the following fields
          pi        * Mixture weights
          mu        * Cell array containing nMix (D x 1) mean vectors
          W         * Cell array containing nMix (D x M) factor loading matrices
          Sigma     * D x D diagonal covariance matrix of noise e
          P,Q       * Cell array containing P and Q matrix (for future use)
          const     * const for computing log-likelihood during scoring (for future use)
          Z         * M x T common factors (one column for each z_i)
     Example:
       Use clean_tel + clean_mic + 15dB_tel + 6dB_tel for training
       tgtcl = load('mat/fw60/male_target-mix_mix_t500_w_1024c.mat');
       tgt06 = load('mat/fw60/male_target-tel-06dB_mix_t500_w_1024c.mat');
       tgt15 = load('mat/fw60/male_target-tel-15dB_mix_t500_w_1024c.mat');
       snrcl = load('../../snr/male/male_target-mix_stats.snr');
       snr06 = load('../../snr/male/male_target-tel-06dB_stats.snr');
       snr15 = load('../../snr/male/male_target-tel-15dB_stats.snr');
       X = [tgtcl.w; tgt15.w; tgt06.w];
       L = [snrcl; snr15; snr06];
       spk_logical = [tgtcl.spk_logical; tgt15.spk_logical; tgt06.spk_logical];
       GPLDAModel = comp_mGPLDA(X, L, spk_logical, 'mat/fw60/male_mix_t500_mgplda_cln-15dB-06dB_1024c.mat', 150, 3);
     Author: M.W. Mak
     Date: Jan 2014
    """

    # Set up path to use GPLDA package
    if platform.system() == 'Windows':
        sys.path.append('D:/so/Matlab/PLDA/BayesPLDA')
    else:
        sys.path.append('~/so/Matlab/PLDA/BayesPLDA')

    # Default training function is mPLDA_train4

    N_ITER = 4
    N_SPK_FAC = n_ev

    # Remove i - vecswith big norm
    X, L, spk_logical = remove_bad_ivec(X, L, spk_logical, 40)

    # Remove speaker with less than 2 utts
    X, L, spk_logical = remove_bad_spks(X, L, spk_logical, 2)

    # Limit the number of speakers and no. of sessions per speakers
    # (for finding the relationship between the no. of speakers in PLDA and performacne)
    # X, L, spk_logical = limit_spks(X, L, spk_logical, 200, 6)

    # Compute WCCN projection matrix and global mean vector. WCNN+lennorm get the best result
    X = X.T     # Convert to column vectors
    projmat1, meanVec1 = wccn(X, spk_logical)

    # Transform i-vector by WCCN projection matrix
    X = np.dot(projmat1.T, (X - np.tile(meanVec1, [1, X.shape[1]])))

    # Perform length normalization
    Xln = (len_norm(X.T)).T

    # LDA+WCCN on length-normalized i-vecs
    Xln, projmat2 = ldawccn(Xln.T, spk_logical, 200)
    Xln = Xln.T

    # Convert BUT's speaker id info (spk_logical) to PLDA equivalent (GaussPLDA)
    PLDA_spkid = BUT2PLDA(spk_logical)

    # # test~!
    # Xln_test = loadmat('Xln.mat')['Xln']
    # Xln = Xln_test

    # Train mPLDA model using the function specified in the function handle
    GPLDAModel = mPLDA_trainfunc(Xln, L, PLDA_spkid, N_ITER, N_SPK_FAC, n_mix)

    GPLDAModel.projmat1 = projmat1
    GPLDAModel.projmat2 = projmat2
    GPLDAModel.meanVec1 = meanVec1

    # Save mGPLDA model
    print('Saving mGPLDA model to %s\n', GPLDA_file)
    # savemat(GPLDA_file, {'GPLDAModel': GPLDAModel})
    GPLDAModel.save_model(GPLDA_file)  # save to .npy file

    return GPLDAModel