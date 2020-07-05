import numpy as np
from scipy.linalg import inv, cholesky, norm, eig
from plda.my_func import lennorm

import warnings
warnings.warn("degree of freedom in cov is set to 0")


class PreProcess:
    def __init__(self, X, spk_ids=None, ndim=None, model=None,):
        self.X = X
        self.spk_ids = spk_ids
        if model is None:
            if spk_ids is None:
                raise ValueError
            else:
                self.unique_ids, self.counts = np.unique(spk_ids, return_counts=True)
        self.ndim = ndim
        self.wccn_matrix = None
        self.whiten_matrix = None
        self.lda_wccn_matrix = None
        self.lda_matrix = None
        self.model = model

    # Define getter function for mu
    @property
    def mu(self):
        if self.model is None:
            return self.X.mean(0)
        else:
            raise NotImplementedError(
                'property mu only meanted to be used in training')

    def demean(self, mu_flag=None):
        if self.model is None:
            self.X -= self.mu
        elif mu_flag == 'mu1':
            self.X -= self.model['mu1']
        elif mu_flag == 'mu2':
            self.X -= self.model['mu2']
        else:
            raise ValueError('no mu has been specified')
        return self

    # Perform Cholesky whitening and get the whitening matrix
    def trans_whiten(self, mu1=None, X_ind=None):
        if self.model is None:
            if X_ind is not None:
                self.whiten_matrix = cholesky(inv(np.cov(self.X_ind.T, ddof=0)), lower=True)
            else:
                self.whiten_matrix = cholesky(inv(np.cov(self.X.T, ddof=0)), lower=True)
        else:
            self.whiten_matrix = self.model['whiten_matrix']
        if mu1 is not None:
            self.X = (self.X - mu1) @ self.whiten_matrix
        else:
            self.X = (self.X - self.X.mean(0)) @ self.whiten_matrix     # Data must be zero-mean
        return self

    def trans_wccn(self):
        if self.model is None:
            self.wccn_matrix = get_wccn(self.X, self.spk_ids, self.unique_ids)
        else:
            self.wccn_matrix = self.model['wccn_matrix']
        self.X[:] = (self.X - self.X.mean(0)) @ self.wccn_matrix
        return self

    def lennorm(self):
        self.X[:] = self.X / norm(self.X, axis=1)[:, None]
        return self

    def trans_lda_wccn(self):
        if self.model is None:
            lda_matrix = get_lda(self.X, self.spk_ids,
                                 self.ndim, self.unique_ids, self.counts)
            wccn_matrix = get_wccn(self.X @ lda_matrix, self.spk_ids, self.unique_ids)
            self.lda_wccn_matrix = lda_matrix @ wccn_matrix
        else:
            self.lda_wccn_matrix = self.model['lda_wccn_matrix']
        self.X = self.X @ self.lda_wccn_matrix
        return self

    def trans_lda(self):
        if self.model is None:
            lda_matrix = get_lda(self.X, self.spk_ids,
                                 self.ndim, self.unique_ids, self.counts)
        else:
            lda_matrix = self.model['lda_matrix']
        self.lda_matrix = lda_matrix
        self.X = self.X @ lda_matrix
        return self


def get_wccn(X, spk_ids, unique_ids=None,):
    return cholesky(inv(_within_group_cov(X, spk_ids, unique_ids,)), lower=True)


def get_lda(X, spk_ids, ndim, unique_ids=None, counts=None):
    cov_within = _within_group_cov(X, spk_ids, unique_ids, )
    cov_between = _between_group_cov(X, spk_ids, unique_ids, counts)
    w, V = eig(a=cov_between, b=cov_within)
    return V[:, np.argsort(-w)[:ndim]]


def _within_group_cov(X, spk_ids, unique_ids=None, group_wight=False):
    if unique_ids is None:
        unique_ids = np.unique(spk_ids)
    ndim = X.shape[-1]
    cov_within = np.zeros((ndim, ndim))
    for unique_id in unique_ids:
        X_homo = X[spk_ids == unique_id]
        if group_wight:
            cov_within += X_homo.shape[0] * np.cov(X_homo.T, ddof=0)
        else:
            cov_within += np.cov(X_homo.T, ddof=0)
    return cov_within / unique_ids.shape[0]


def _between_group_cov(X, spk_ids, unique_ids=None, counts=None, group_wight=False):
    if unique_ids is None:
        unique_ids, counts = np.unique(spk_ids, return_counts=True)
    n_uniq = unique_ids.shape[0]
    ndim = X.shape[-1]
    mu_within = np.zeros((n_uniq, ndim))
    for i, unique_id in enumerate(unique_ids):
        mu_within[i] = X[spk_ids == unique_id].mean(0)
    aux = mu_within - X.mean(0)
    if group_wight:
        return (counts * aux.T) @ aux
    else:
        return aux.T @ aux


# Preprocess i-vectors based on the preprocessing mode. Note that the function is
# different from the preprocess() function in batch_plda_scoring.py
def preprocess(x, model, mode='lennorm', mu_outd=None):
    mu2 = model['mu2'] if mu_outd is None else mu_outd
    if mode == 'lennorm':
        x = lennorm(x) - mu2
    elif mode == 'wccn+lennorm+wccn':
        y = (x - model['mu1']) @ model['wccn_matrix1']
        x = lennorm(y) @ model['wccn_matrix2'] - mu2
    elif mode == 'none':
        # Note that it is wrong because we have to subtract the global mean
        pass
    else:
        raise NotImplementedError
    return x


