import numpy as np
# import cupy as np
from len_norm_ import len_norm

def Mahaldist(x, mu, Icov):
    """
     Return the Mahalanobis distance between x and mu with covariance Sigma
     Both x and mu are col vectors
    """

    temp = x - mu
    md = np.dot(np.dot(temp.T, Icov), temp)
    return md

def mPLDA_GroupScoring5(mPLDAModel, Xs, xt):
    """
     function [scores, clusterID] = mPLDA_GroupScoring1(mPLDAModel, Xs, xt)
     Implement the SNR-independent mixture of PLDA scoring. This function should
     work with mPLDA_train5_.py

     This function is for opt.mode = 'scravg' and 'ivcsnravg' in snr3_score_gplda_w_.py.
     When mode = 'icvsnravg', the SNR within the same group must be averaged and the
     length of Ls must be equal to the number of SNR groups.

       Input:
           mPLDAModel     - mPLDA model structure
           Xs             - Matrix containing a set of column i-vectors of speaker s
           xt             - Second un-normalized i-vector (column vec)
       Output:
           scores         - PLDA scores (unnormalized) of Xs and xt
     Author: M.W. Mak
     Date: Aug 2015
    """

    n_vecs = Xs.shape[1]
    Xs = np.dot(mPLDAModel.projmat1.T, Xs - np.tile(mPLDAModel.meanVec1, [1, n_vecs]))
    xt = np.dot(mPLDAModel.projmat1.T, xt - mPLDAModel.meanVec1)
    Xs = len_norm(Xs.T).T
    xt = (len_norm(xt.T)).T
    Xs = np.dot(mPLDAModel.projmat2.T, Xs)
    xt = np.dot(mPLDAModel.projmat2.T, xt)

    # Extract paras from model structure
    varphi = mPLDAModel.varphi
    m = mPLDAModel.m
    Icov = mPLDAModel.Icov
    Icov2 = mPLDAModel.Icov2
    logDetCov = mPLDAModel.logDetCov
    logDetCov2 = mPLDAModel.logDetCov2
    V = mPLDAModel.V
    K = len(varphi)

    # Precompute likelihood of test i-vector for speed
    sum3 = 0
    expterm = np.zeros(shape=(K, 1), dtype=np.float64)
    for k in range(K):
        detterm = 0.5 * logDetCov[0][k]
        expterm[k, 0] = np.exp(-0.5 * Mahaldist(xt, m[0][k], Icov[0][k]) - detterm)
        sum3 = sum3 + varphi[k, 0] * expterm[k, 0]

    assert sum3 > 0, 'Divided by zero (sum3) in mPLDA_GroupScoring_.py'

    # Compute log-likelihood score
    scores = np.zeros(shape=(Xs.shape[1], 1), dtype=np.float64)
    for s in range(len(scores)):
        sum1 = 0
        sum2 = 0
        xs = Xs[:, s].reshape(-1, 1)
        for ks in range(K):
            for kt in range(K):
                sum1 = sum1 + varphi[ks][0] * varphi[kt][0] * \
                   np.exp(-0.5 * Mahaldist(np.vstack((xs, xt)), np.vstack((m[0][ks], m[0][kt])), Icov2[ks][kt]) - 0.5 * logDetCov2[ks][kt])
        for k in range(K):
            detterm = 0.5 * logDetCov[0][k]
            sum2 = sum2 + varphi[k][0] * np.exp(-0.5 * Mahaldist(xs, m[0][k], Icov[0][k]) - detterm)
        assert sum1 > 0, 'Sum1 is 0 in mPLDA_GroupScoring_.py'
        assert sum2 > 0, 'Divided by zero (sum2) in mPLDA_GroupScoring_.py'
        scores[s][0] = np.log(sum1) - (np.log(sum2) + np.log(sum3))
    return scores