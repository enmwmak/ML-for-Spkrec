import numpy as np
# import cupy as np
from scipy.linalg import eigh
from scipy.linalg import cholesky

def ldawccn(w, spk_logical, n_ev):
    """
     Estimate the LDA+WCCN matrix  from w and spk_logical.

     Rank of WCCN and LDA
     S_w = sum_s sum_i (x_si – m_s)(x_si – m_s)’
     Rank(S_w) = S * rank(cov(x_si))     if rank(cov(x_si) > 1
     Rank(S_w) = S                       if rank(cov(x_si) = 1
    """

    _, _, spk_ids = np.unique(spk_logical, return_index=True, return_inverse=True)  # spk_ids contains indexes to unique speakers
    nf = w.shape[1]
    Sw = np.zeros(shape=(nf, nf), dtype=np.float64)
    Sb = np.zeros(shape=(nf, nf), dtype=np.float64)
    mu = np.mean(w, axis=0).reshape((1, w.shape[1]))
    for ii in np.unique(spk_ids):
        spk_sessions = np.where(spk_ids == ii)[0]
        ws = w[spk_sessions, :]
        Sw = Sw + np.cov(ws.T, bias=True)
        mu_s = np.mean(ws, axis=0)
        Sb = Sb + np.dot((mu_s-mu).T, mu_s-mu)

    # Find the n_ev largest eigenvectors and eigenvalues of AV=Lambda BV, i.e.,
    # find the eigenvectors of inv(Sw)*Sb
    _, V = eigh(Sb, b=Sw, subset_by_index = [nf-n_ev, nf-1])
    V = np.fliplr(V)

    # Project the total factors to a reduced space using the LDA projection matrix
    lda_y = np.dot(w, V)

    # Find WCCN
    n_spks = len(np.unique(spk_ids))
    Ws = np.zeros(shape=(n_ev, n_ev), dtype=np.float64)
    for ii in np.unique(spk_ids):
        spk_sessions = np.where(spk_ids == ii)[0]
        Ws = Ws + np.dot(np.cov(lda_y[spk_sessions, :].T, bias=True), len(spk_sessions))
    Winv = np.linalg.inv(Ws/n_spks)
    B = cholesky(Winv, lower=True)  # Compute projection matrix so that BB'=inv(W)

    # Return projected vectors (row vecs) in X and LDA+WCCN projection matrix
    projmat = np.dot(V, B)
    X = np.dot(w, projmat)
    return X, projmat