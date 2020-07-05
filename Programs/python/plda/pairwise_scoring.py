"""
Compute pairwise PLDA score matrix, distance matrix, and similarity matrix.
Save and load the score matrix.

M.W. Mak, Nov. 2017
"""

import h5py as h5
import numpy as np
from plda.preprocess import preprocess


# Save PLDA score matrix to .h5 file
def save_scoremat(filename, scoremat):
    print('Saving %s' % filename)
    with h5.File(filename) as f:
        f['scr_mat'] = scoremat


# Save PLDA score matrix to .h5 file
def load_scoremat(filename):
    print('Loading %s' % filename)
    with h5.File(filename) as f:
        scoremat = f['scr_mat'][:]
    return scoremat


# Convert to similarity matrix and distance matrix
def score2sim(score_mat, sigma=1):
    s_amax = np.max(np.abs(score_mat))
    dist_mat = s_amax - score_mat
    np.fill_diagonal(dist_mat, 0.0)
    return np.exp(-1 * np.square(dist_mat)/(2 * sigma * sigma)), dist_mat


# Compute pairwise scores, ignoring self-comparisons
def get_plda_matrix(X, model, mode='lennorm', partial=False):
    n_ivecs = X.shape[0] if partial is False else np.min([200, X.shape[0]])
    X = preprocess(X, model, mode)
    scores = np.zeros((n_ivecs, n_ivecs))
    for i in range(n_ivecs):
        print('Processing %d of %d' % (i, n_ivecs), end='\r')
        for j in range(i+1, n_ivecs):
            scores[i, j] = get_score(X[i, :], X[j, :], model=model)
            scores[j, i] = scores[i, j]
    return scores


# Compute the PLDA score of two preprocessed i-vectors (xs, xt)
# scores = PLDAModel.const + 0.5*(xs'*Q*xs + xt'*Q*xt + 2*xs'*P*xt);
def get_score(xs, xt, model):
    P = model['P']
    Q = model['Q']
    return model['const'] + 0.5*(xs.T @ Q @ xs.T + xt @ Q @ xt.T + 2*xs @ P @ xt.T)


# Compute pairwise scores, block-implemented.
def get_plda_matrix_block(X, model, mode='lennorm', partial=False):
    n_ivecs = X.shape[0] if partial is False else np.min([200, X.shape[0]])
    X = preprocess(X, model, mode)

    QX = X @ model['Q']
    xQxt_mat = np.einsum('ij,ij->i', X, QX)  # Diag(X * Q * X') for row-vectors in X
    PX = X @ model['P']
    xPxt = X @ PX.T
    xPxt = xPxt + 0.5 * (xQxt_mat + xQxt_mat[:, None]) + model['const']

    scores = np.triu(xPxt, k=1)         # Get upper triangular matrix without diagonal
    scores = scores + scores.T          # Copy to lower trianglular matrix

    return scores[0:n_ivecs, 0:n_ivecs]
