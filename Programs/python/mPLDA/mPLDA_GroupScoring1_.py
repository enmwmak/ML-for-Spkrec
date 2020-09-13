import numpy as np
from len_norm_ import len_norm
from posterior1G_ import posterior1G
from posterior2G_ import posterior2G

def Mahaldist(x, mu, Icov):
    """
    Return the Mahalanobis distance between x and mu with covariance Sigma
    Both x and mu are col vectors
    """

    temp = x - mu
    md = np.dot(np.dot(temp.T, Icov), temp)
    return md

def mPLDA_GroupScoring1(mPLDAModel, Xs, xt, Ls, lt):
    """
     function [scores, clusterID] = mPLDA_GroupScoring1(mPLDAModel, Xs, Ls, xt, lt)
     Implement the mixture of PLDA scoring in
     M.W. Mak, SNR-Dependent Mixture of PLDA for Noise Robust Speaker Veriﬁcation, Intespeech2014.
     The test i-vector is scored against each of the target-speaker's i-vectors.
     No averaging is applied to the target-speaker's i-vectors.
     This function is for opt.mode = 'scravg' and 'ivcsnravg' in snr3_score_gplda_w_.py.
     When mode = 'icvsnravg', the SNR within the same group must be averaged and the
     length of Ls must be equal to the number of SNR groups.

       Input:
           mPLDAModel     - mPLDA model structure
           Xs             - Matrix containing a set of column i-vectors of speaker s
           xt             - Second un-normalized i-vector (column vec)
           Ls             - Length or SNR of utterances in Xs
           lt             - Length or SNR of utterance in xt
       Output:
           scores         - PLDA scores (unnormalized) of Xs and xt
     Author: M.W. Mak
     Date: June 2014
     Update May 2015 Mak: The likelihood of test i-vector is computed outside onces only,
                          which speed up the computation of LLR when the number of target
                          speaker i-vecs is large.
    """

    n_vecs = Xs.shape[1]
    Xs = np.dot(mPLDAModel.projmat1.T, Xs - np.tile(mPLDAModel.meanVec1, [1, n_vecs]))
    xt = np.dot(mPLDAModel.projmat1.T, xt - mPLDAModel.meanVec1)
    Xs = len_norm(Xs.T).T
    xt = (len_norm(xt.T)).T
    Xs = np.dot(mPLDAModel.projmat2.T, Xs)
    xt = np.dot(mPLDAModel.projmat2.T, xt)

    # Extract paras from model structure
    pi = mPLDAModel.pi
    mu = mPLDAModel.mu
    sigma = mPLDAModel.sigma
    m = mPLDAModel.m
    Icov = mPLDAModel.Icov
    Icov2 = mPLDAModel.Icov2
    logDetCov = mPLDAModel.logDetCov
    logDetCov2 = mPLDAModel.logDetCov2
    V = mPLDAModel.V
    K = len(pi)

    # Precompute likelihood of test i-vector for speed
    posty_lt = posterior1G(lt, pi, mu, sigma)
    sum3 = 0
    expterm = np.zeros(shape=(K, 1), dtype=np.float64)
    for k in range(K):
        detterm = 0.5 * logDetCov[0][k]
        expterm[k, 0] = np.exp(-0.5 * Mahaldist(xt, m[0][k], Icov[0][k]) - detterm)
        sum3 = sum3 + posty_lt[0, k] * expterm[k, 0]

    assert sum3 > 0, 'Divided by zero (sum3) in mPLDA_GroupScoring_.py'

    # compute log-likelihood score
    scores = np.zeros(shape=(len(Ls), 1), dtype=np.float64)
    for s in range(len(Ls)):
        sum1 = 0
        sum2 = 0
        xs = Xs[:, s].reshape(-1, 1)
        ls = Ls[s]
        posty = posterior2G(ls, lt, pi, mu, sigma)
        for ks in range(K):
            for kt in range(K):
                sum1 = sum1 + posty[ks, kt] * np.exp(-0.5 * Mahaldist(np.vstack((xs, xt)), np.vstack((m[0][ks], m[0][kt])), Icov2[ks][kt]) - 0.5 * logDetCov2[ks][kt])
        posty_ls = posterior1G(ls, pi, mu, sigma)
        for k in range(K):
            detterm = 0.5 * logDetCov[0][k]
            sum2 = sum2 + posty_ls[0, k] * np.exp(-0.5 * Mahaldist(xs, m[0][k], Icov[0][k]) - detterm)
        assert sum1 > 0, 'Sum1 is 0 in mPLDA_GroupScoring1_.py'
        assert sum2 > 0, 'Divided by zero (sum2) in mPLDA_GroupScoring1_.py'
        scores[s][0] = np.log(sum1) - (np.log(sum2) + np.log(sum3))
    return scores