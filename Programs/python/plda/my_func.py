import numpy as np
from scipy import linalg as la
from scipy.stats import multivariate_normal


class GaussianDistribution:
    def __init__(self, mean=None, cov=None):
        if mean is None:
            self.mean = np.zeros(cov.shape[-1])
        else:
            self.mean = mean
        self.cov = cov

    def log_p(self, x):
        return multivariate_normal.logpdf(x, self.mean, self.cov)

    def sample(self, size=None):
        return np.random.multivariate_normal(self.mean, self.cov, size)


def lennorm(X):
    return X / la.norm(X, axis=1)[:, None]


def log_det4psd(sigma):
    return 2 * np.sum(np.log(np.diag(la.cholesky(sigma))))


def ravel_dot(X, Y):
    return X.ravel() @ Y.ravel()
