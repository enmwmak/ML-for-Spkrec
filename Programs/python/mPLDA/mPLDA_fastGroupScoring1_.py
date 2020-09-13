import numpy as np
from len_norm_ import len_norm
from posterior1G_ import posterior1G
from posterior2G_ import posterior2G

def mPLDA_fastGroupScoring1(mPLDAModel, Xs, xt, Ls, lt):
    """
     function scores = mPLDA_fastGroupScoring1(mPLDAModel, Xs, Ls, xt, lt)
     Implement fast scoring of the mixture of PLDA scoring in
     M.W. Mak, SNR-Dependent Mixture of PLDA for Noise Robust Speaker Veriﬁcation, Intespeech2014.
     The test i-vector is scored against each of the target-speaker's i-vectors.
     No averaging is applied to the target-speaker's i-vectors.
     This function is for opt.mode = 'scravg' and 'ivcsnravg' in snr3_score_gplda_w_.py.
     When mode = 'icvsnravg', the SNR within the same group must be averaged and the
     length of Ls must be equal to the number of SNR groups.

     Note: To achieve fast scoring, only the Gaussian with largest posterior will be considered.
       Input:
           mPLDAModel     - mPLDA model structure
           Xs             - Matrix containing a set of column i-vectors of speaker s
           xt             - Second un-normalized i-vector (column vec)
           Ls             - Length of utterances in Xs
           lt             - Length of utterances in xt
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

    # Find the index to the maximum posterior test SNR
    posty_lt = posterior1G(lt, pi, mu, sigma)
    kt1 = np.argmax(posty_lt)

    # Compute log-likelihood score
    scores = np.zeros(shape=(len(Ls), 1), dtype=np.float64)
    for s in range(len(Ls)):
        xs = Xs[:, s].reshape(-1, 1)
        ls = Ls[s]
        posty = posterior2G(ls, lt, pi, mu, sigma)
        ks2, kt2 = np.where(posty == np.max(posty))
        posty_ls = posterior1G(ls, pi, mu, sigma)
        ks1 = np.argmax(posty_ls)
        Pst = mPLDAModel.P[ks2, kt2][0]
        Qst = mPLDAModel.Q[ks2, kt2][0]
        Qts = mPLDAModel.Q[kt2, ks2][0]
        ms = mPLDAModel.m[0, ks2][0]
        mt = mPLDAModel.m[0, kt2][0]
        const = mPLDAModel.const[ks2, kt2][0]
        scores[s][0] = np.log(posty[ks2, kt2]) - np.log(posty_ls[0, ks1]) - np.log(posty_lt[0, kt1]) \
                       + 0.5 * np.dot(np.dot(xs.T, Qst), (xs + 2 * ms)) + 0.5 * np.dot(np.dot(xt.T, Qts), (xt + 2 * mt)) \
                       + np.dot(np.dot(xs.T, Pst), (xt + mt)) + np.dot(np.dot(xt.T, Pst.T), ms) \
                       + const
    return scores

