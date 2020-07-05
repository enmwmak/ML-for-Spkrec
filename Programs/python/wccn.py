#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 18 11:25:46 2019

@author: mwmak

Implementation of within-class covariance normalization (WCCN)
"""

import numpy as np
import pickle
from scipy.linalg import inv, cholesky

class Transform:
    def __init__(self):
        self.trained = False

    def fit(self, X, spk_ids):
        raise NotImplementedError()

    def transform(self, X):
        raise NotImplementedError()

    def fit_transform(self, X, spk_ids=None):
        if self.trained:
            return self.transform(X)
        else:
            return self.fit(X, spk_ids).transform(X)

    def check_if_model_exist(self):
        if self.trained:
            raise ValueError('a trained model already exist.')
        else:
            self.trained = True


class WCCN(Transform):
    def __init__(self):
        super().__init__()

    def fit(self, X, y, epsilon=0.001):
        self.check_if_model_exist()
        wcc_mat = self.within_class_cov(X, y, class_weight=False)
        wcc_mat = wcc_mat + np.eye(X.shape[-1])*epsilon
        self.proj = cholesky(inv(wcc_mat), lower=True)
        return self
        
    def transform(self, X):
        return X @ self.proj

    def save_model(self, filename):
        with open(filename, 'wb') as f:  # Overwrites any existing file.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod        
    def load_model(filename):
        with open(filename, 'rb') as f:
            return (pickle.load(f))

    def within_class_cov(self, X, y, class_weight=False):
        unique_ids = np.unique(y)
        ndim = X.shape[-1]
        cov_within = np.zeros((ndim, ndim))
        for unique_id in unique_ids:
            X_homo = X[y == unique_id]
            if class_weight:
                cov_within += X_homo.shape[0] * np.cov(X_homo.T, ddof=0)
            else:
                cov_within += np.cov(X_homo.T, ddof=0)
        return cov_within / unique_ids.shape[0]

