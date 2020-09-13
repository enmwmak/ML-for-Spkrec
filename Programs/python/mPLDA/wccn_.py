import numpy as np
# import cupy as np
from scipy.linalg import cholesky
def wccn(X, spk_logical):
    """
     Compute WCCN projection matrix and glogal mean i-vector (col vec)
     X contains i-vectors in columns
    """

    # Get zero mean vectors
    meanVec = np.mean(X, axis =1).reshape(X.shape[0], 1)

    # Estmate projection matrix B
    _, _, spk_ids = np.unique(spk_logical, return_index=True, return_inverse=True) # spk_ids contains indexes to unique speakers
    numSpks = len(np.unique(spk_ids))
    dim = X.shape[0]
    Ws = np.zeros(shape=(dim, dim), dtype=np.float64)
    for ii in np.unique(spk_ids):
        spk_sessions = np.where(spk_ids == ii)[0]
        Ws = Ws + np.cov(X[:, spk_sessions], bias=True)  # *length(spk_sessions); # Multiplied by length get better result
    Winv = np.linalg.inv(Ws/numSpks)
    B = cholesky(Winv, lower=True)    # linalg.cholesky(a) returns a lower triangular matrix

    # Perform WCCN
    # Xwccn = w * B;
    return B, meanVec