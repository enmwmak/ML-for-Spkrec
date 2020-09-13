import numpy as np
# import cupy as np
from scipy import linalg as la

def len_norm(X):
    """
     def len_norm(X):
         # Return length normalized vectors in rows of Xln
         Xln = np.zeros(shape=X.shape, dtype=np.float64)
         for i in range(X.shape[0]):
             Xln[i, :]= X[i, :]/np.linalg.norm(X[i, :])
         return Xln
    """

    return X / la.norm(X, axis=1)[:, None]