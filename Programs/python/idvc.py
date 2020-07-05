#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 12 19:05:07 2019

Implementation of Inter-dataset variability compensation (IDVC)

"""

import numpy as np
from sklearn.decomposition import PCA
import scipy
import pickle
from wccn import WCCN

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


class IDVC(Transform):
    def __init__(self, n_components=None, whiten=False, whiten_method=None):
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten
        self.whiten_method = whiten_method             # Could be 'PCA' or 'WCCN'

    def fit(self, X, dom_lbs, spk_lbs=None):
        self.check_if_model_exist()
        means = []
        for dom_lbs_uni in np.unique(dom_lbs):
            means.append(X[dom_lbs == dom_lbs_uni].mean(0))
        means = np.stack(means)
        means = means - means.mean(0)
        _, _, Vh = scipy.linalg.svd(means, full_matrices=False)
        self.proj = np.eye(X.shape[-1]) - Vh.T @ Vh
        if self.n_components is not None and self.whiten_method == 'PCA':
            self.wht_obj = PCA(n_components=self.n_components, whiten=self.whiten).fit(X @ self.proj)
        elif self.whiten_method == 'WCCN':
            self.wht_obj = WCCN().fit(X @ self.proj, spk_lbs)
        return self

    def transform(self, X):
        if self.whiten is True or self.n_components is not None:
            return self.wht_obj.transform(X @ self.proj)
        else:
            return X @ self.proj

    def save_model(self, filename):
        with open(filename, 'wb') as f:  # Overwrites any existing file.
            pickle.dump(self, f, pickle.HIGHEST_PROTOCOL)
    
    @staticmethod        
    def load_model(filename):
        with open(filename, 'rb') as f:
            return (pickle.load(f))
        