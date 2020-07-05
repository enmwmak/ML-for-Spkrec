import numpy as np
from numpy.random import randn
import h5py as h5
from plda.preprocess import PreProcess
import scipy.linalg as linalg
from plda.my_func import GaussianDistribution as Gauss
from scipy.linalg import eigvalsh, inv, norm
from plda.my_func import lennorm
import matplotlib.pyplot as plt


class Plda:
    def __init__(self, model_file, datafiles=None, indomain_datafiles=None,
                 n_fac=150, lda_dim=None,
                 prep_mode='wccn+lennorm+lda+wccn',
                 n_iter=10, utts_cutoff=None,
                 norm_cutoff=None,
                 W_init=None, mu_int=None, sigma_init=None, run_toy=False,
                 whiten_matrix=None, mu2=None, mu1=None, mu_outd=None):
        self.n_fac = n_fac
        self.prep_mode = prep_mode
        self.n_iter = n_iter
        self.datafiles = datafiles
        self.indomain_datafiles = indomain_datafiles
        if prep_mode == 'wccn+lennorm+lda+wccn':
            self.lda_dim = lda_dim
        else:
            self.lda_dim = None
        self.utts_cutoff = utts_cutoff
        self.norm_cutoff = norm_cutoff
        self.model_file = model_file
        self.W_init = W_init
        self.mu_init = mu_int
        self.sigma_init = sigma_init
        self.llh_lst = [-np.inf]
        self.wccn_matrix1, self.lda_wccn_matrix, self.wccn_matrix2 = None, None, None
        self.W, self.sigma = None, None
        self.run_toy = run_toy
        if self.run_toy:
            self.n_fac, self.prep_mode, self.norm_cutoff, \
                self.utts_cutoff, self.lda_dim = 3, None, None, None, None
        self.whiten_matrix = whiten_matrix
        self.mu2 = mu2              # Mean of training i-vectors after lennorm
        self.mu1 = mu1              # Mean of training i-vectors after WCCN-whitening
        self.mu_outd = mu_outd              # Mean of out-of-domain training data.
        self.lda_matrix = None
        self.mu_ind = 0  # Mean of in-domain data
        self.wccn_matrix = None

    @property
    def prec(self):
        return inv(self.sigma)

    def fit(self, X=None, spk_ids=None):
        if X is None:
            X, spk_ids, X_ind = self.load_data()
        if X_ind is not None:
            self.mu_ind = X_ind.mean(0)
        X, spk_ids = self.process_traning_data(X, spk_ids, X_ind=None)
        print('No. of training vectors = %d' % len(X))
        self.init_params(X)
        xstat = _suff_xstats(X, spk_ids)
        for i_iter in range(self.n_iter):
            print('Epoch %d/%d: ' % (i_iter+1, self.n_iter), end='')
            zstat = self.e_step(xstat)
            llh = self.comp_llh(xstat, zstat, mode='elbo')
            self._display_llh(llh)
            self.m_step(xstat, zstat)
        self.save_model()
        return self

    def load_data(self):
        X, spk_ids, X_ind = [], [], []
        for file in self.datafiles:
            print('Loading %s' % file)
            with h5.File(file) as f:
                X.append(f['X'][:])
                spk_ids.append(f['spk_ids'][:])
        X = np.concatenate(X, axis=0)
        spk_ids = np.concatenate(spk_ids, axis=0)
        if self.indomain_datafiles is not None:
            for file in self.indomain_datafiles:
                with h5.File(file) as f:
                    X_ind.append(f['X'][:])
            X_ind = np.concatenate(X_ind, axis=0)
        else:
            X_ind = None
        return X, spk_ids, X_ind

    def init_params(self, X):
        n_fea = (X.shape[1] if self.lda_dim is None
                 else self.lda_dim)
        # mu2 cann't be init here
        # self.mu2 = (X.mean(0) if self.mu_init is None
        #             else self.mu_init)
        self.W = (randn(self.n_fac, n_fea) if self.W_init is None
                  else self.W_init)
        self.sigma = (abs(randn()) * np.eye(n_fea) if self.sigma_init is None
                      else self.sigma_init)

    def process_traning_data(self, X, spk_ids, X_ind=None, spk_ids_ind=None):
        if self.utts_cutoff or self.norm_cutoff is not None:
            X, spk_ids = self._remove_bad_ivcs(X, spk_ids)

        print(self.prep_mode)

        # If in-domain data are provided, save the mean and WCCN estimated by in-domain data
        if self.prep_mode == 'lda+lennorm':
            X = X - X.mean(0)
            prep = PreProcess(X=X, spk_ids=spk_ids, ndim=self.lda_dim).trans_lda()
            self.mu1 = prep.mu
            self.lda_matrix = prep.lda_matrix
            prep.lennorm()
            self.mu2 = prep.mu
            X = prep.X

        elif self.prep_mode == 'wccn+lennorm+wccn':
            prep = PreProcess(X=X, spk_ids=spk_ids, ndim=self.lda_dim).trans_wccn()
            self.wccn_matrix1, self.mu1 = prep.wccn_matrix, prep.mu
            prep.lennorm().trans_wccn()
            self.wccn_matrix2, self.mu2 = prep.wccn_matrix, prep.mu
            X = prep.X
            if X_ind is not None:
                prep = PreProcess(X=X_ind, spk_ids=spk_ids_ind, ndim=self.lda_dim).trans_wccn()
                self.wccn_matrix1, self.mu1 = prep.wccn_matrix, prep.mu
                prep.lennorm().trans_wccn()
                self.wccn_matrix2, self.mu2 = prep.wccn_matrix, prep.mu

        elif self.prep_mode == 'wccn+lennorm+lda+wccn':
            prep = PreProcess(X=X, spk_ids=spk_ids, ndim=self.lda_dim).trans_wccn()
            self.wccn_matrix1, self.mu1 = prep.wccn_matrix, prep.mu
            prep.lennorm().trans_lda_wccn()
            self.lda_wccn_matrix, self.mu2 = prep.lda_wccn_matrix, prep.mu
            X = prep.demean().X
            if X_ind is not None:
                prep = PreProcess(X=X_ind, spk_ids=spk_ids_ind, ndim=self.lda_dim).trans_wccn()
                self.wccn_matrix1, self.mu1 = prep.wccn_matrix, prep.mu
                prep.lennorm().trans_lda_wccn()
                self.lda_wccn_matrix, self.mu2 = prep.lda_wccn_matrix, prep.mu

        elif self.prep_mode == 'lennorm':
            X = lennorm(X)
            self.mu_outd = X.mean(0)            # Global mean of out-of-domain data
            if X_ind is not None:
                X_ind = lennorm(X_ind)
            self.mu2 = X.mean(0) if X_ind is None else X_ind.mean(0)
            X = X - X.mean(0)           # Make sure training vectors for PLDA have zero mean

        elif self.prep_mode == 'given_mu2+lennorm':
            prep = PreProcess(X=X, spk_ids=spk_ids)
            X = prep.lennorm().X
            X -= self.mu2

        elif self.prep_mode == 'wccn+lennorm':
            prep = PreProcess(X=X, spk_ids=spk_ids).trans_wccn()
            self.wccn_matrix = prep.wccn_matrix
            self.mu2 = prep.lennorm().mu
            X = prep.demean().X

        elif self.prep_mode == 'whiten+lennorm':
            if X_ind is not None:
                self.mu1 = X_ind.mean(0)
                prep = PreProcess(X=X_ind, spk_ids=spk_ids).trans_whiten().lennorm()
                self.whiten_matrix = prep.whiten_matrix
                self.mu2 = prep.mu
                X = lennorm((X - self.mu1) @ self.whiten_matrix) - self.mu2
            else:
                self.mu1 = X.mean(0)        # mu1 should be the global mean before whitening
                prep = PreProcess(X=X, spk_ids=spk_ids).trans_whiten().lennorm()
                self.whiten_matrix = prep.whiten_matrix
                self.mu2 = prep.mu
                X = lennorm((X - self.mu1) @ self.whiten_matrix) - self.mu2

        elif self.prep_mode is None:
            self.mu2 = X.mean(0)
            X -= self.mu2
        elif self.prep_mode == 'none':
            self.mu2 = X.mean(0) if X_ind is None else X_ind.mean(0)
            self.mu_outd = X.mean(0)
            X -= self.mu_outd
        else:
            raise NotImplementedError
        return X, spk_ids

    def _remove_bad_ivcs(self, X, idens):
        mask = []
        if self.utts_cutoff is not None:
            unique_ids, idx_inv, counts = np.unique(idens, return_counts=True, return_inverse=True)
            mask.append(counts[idx_inv] >= self.utts_cutoff)
        if self.norm_cutoff is not None:
            X_norm = norm(X, axis=1)
            mask.append(X_norm > self.norm_cutoff)
        mask_comb = np.logical_or.reduce(mask)
        print('remove bad i-vectors')
        return X[mask_comb], idens[mask_comb]

    def e_step(self, x):
        WtP = self.prec @ self.W.T
        WtPW = self.W @ WtP
        n_id = len(x['ns_obs'])
        mu_post = np.zeros((n_id, self.n_fac))
        sigma_post = np.zeros((n_id, self.n_fac, self.n_fac))
        for i_id, (X_homo_sum, n_ob) in enumerate(zip(x['homo_sums'], x['ns_obs'])):
            sigma_post[i_id] = inv(np.eye(self.n_fac) + n_ob * WtPW)
            mu_post[i_id] = X_homo_sum @ WtP @ sigma_post[i_id]
        mu_mom2s = np.einsum('Bi,Bj->Bij', mu_post, mu_post) + sigma_post
        return {'mom1s': mu_post, 'mom2s': mu_mom2s}

    def m_step(self, x, z):
        z_mom2s_sum = np.einsum('B,Bij->ij', x['ns_obs'], z['mom2s'])
        xz_cmom = z['mom1s'].T @ x['homo_sums']
        self.W = inv(z_mom2s_sum) @ xz_cmom
        self.sigma = (x['mom2'] - xz_cmom.T @ self.W) / x['ns_obs'].sum()

    def comp_llh(self, xstat, zstat, mode='elbo', dev=None, spk_ids=None,):
        if mode == 'elbo':
            llh = self.elbo(xstat, zstat)
        else:
            llh = exact_marginal_llh(
                dev=dev, idens=spk_ids, W=self.W, sigma=self.sigma,)
        return llh

    def elbo(self, xstat, zstat):
        WtPW = self.W @ self.prec @ self.W.T
        return - _ce_cond_xs(xstat, zstat, self.W, self.prec) \
               - _ce_prior(zstat) \
               + _entropy_q(xstat['ns_obs'], WtPW)

    def _display_llh(self, llh):
        self.llh_lst.append(llh)
        if self.llh_lst[-2] == -np.inf:
            print('llh = {:.4f} increased inf'.format(llh))
        else:
            margin = self.llh_lst[-1] - self.llh_lst[-2]
            change_percent = 100 * np.abs(margin / self.llh_lst[-2])
            print('llh = {:.4f} {} {:.4f}%'.format(
                llh, 'increased' if margin > 0 else 'decreased', change_percent,))

    def save_model(self):
        self._comp_pq()
        items = [
            ('W', self.W),
            ('P', self.P),
            ('Q', self.Q),
            ('const', self.const),
            ('sigma', self.sigma),
            ('mu_outd', self.mu_outd),
            ('mu1', self.mu1),
            ('mu2', self.mu2),
            ('wccn_matrix1', self.wccn_matrix1),
            ('wccn_matrix2', self.wccn_matrix2),        # Not None for prep_mode='wccn+lennorm+wccn'
            ('lda_wccn_matrix', self.lda_wccn_matrix),
            ('whiten_matrix', self.whiten_matrix),
            ('prep_mode', self.prep_mode),
            ('mu_ind', self.mu_ind),
            ('lda_matrix', self.lda_matrix)
        ]
        with h5.File(self.model_file, 'w') as f:
            for name, val in items:
                if val is None:
                    continue
                f[name] = val
            print('save model to {}'.format(self.model_file))

    def _comp_pq(self):
        sig_ac = self.W.T @ self.W
        sig_tot = sig_ac + self.sigma
        prec_tot = inv(sig_tot)
        aux = inv(sig_tot - sig_ac @ prec_tot @ sig_ac)
        B0 = np.zeros_like(self.sigma)
        M1 = np.block([[sig_tot, sig_ac], [sig_ac, sig_tot]])
        M2 = np.block([[sig_tot, B0], [B0, sig_tot]])
        self.P = aux @ sig_ac @ prec_tot
        self.Q = prec_tot - aux
        self.const = 0.5 * (-log_det4psd(M1) + log_det4psd(M2))


def _suff_xstats(X, spk_ids):
    X -= X.mean(0)
    mom2 = X.T @ X
    unique_ids, ns_obs = np.unique(spk_ids, return_counts=True)
    homo_sums = np.zeros((unique_ids.shape[0], X.shape[-1]))
    for i_id, unique_id in enumerate(unique_ids):
        homo_sums[i_id] = np.sum(X[spk_ids == unique_id], axis=0)
    return {'mom2': mom2, 'homo_sums': homo_sums, 'ns_obs': ns_obs}

_LOG_2PI = np.log(2 * np.pi)


def _ce_cond_xs(x, z, W, prec):
    dim = prec.shape[-1]
    N = x['ns_obs'].sum()
    xy_cmom = W.T @ z['mom1s'].T @ x['homo_sums']
    z_mom2s_wsum = np.einsum('B,Bij->ij', x['ns_obs'], z['mom2s'])
    dev_mom2 = x['mom2'] - xy_cmom - xy_cmom.T + W.T @ z_mom2s_wsum @ W
    return 0.5 * (N * dim * _LOG_2PI
                  - N * log_det4psd(prec)
                  + ravel_dot(dev_mom2, prec))


def _ce_prior(z):
    n_ids, dim = z['mom1s'].shape
    return 0.5 * (n_ids * dim * _LOG_2PI
                  + np.einsum('Bii->', z['mom2s']))


def _entropy_q(ns_obs, WtPW):
    n_ids = len(ns_obs)
    zdim = WtPW.shape[0]
    # due to the special form of posterior co logdet can be greatly simplified
    eigvals = np.outer(ns_obs, eigvalsh(WtPW)) + 1
    log_det_sum = np.sum(np.log(1 / eigvals))
    return 0.5 * (n_ids * zdim * _LOG_2PI
                  + log_det_sum
                  + n_ids * zdim)


def log_det4psd(sigma):
    return 2 * np.sum(np.log(np.diag(linalg.cholesky(sigma))))


def ravel_dot(X, Y):
    return X.ravel() @ Y.ravel()


def exact_marginal_llh(dev, idens, W, sigma):
    # this is very computation intensive op should only be used to
    # check whether low-bound is correct on toy data
    # stake mu is 0, diag of cov is sigma + WWt, off-diag is WWt
    llh = 0.0
    unique_ids = np.unique(idens)
    for unique_id in unique_ids:
        dev_homo = dev[idens == unique_id]
        cov = _construct_marginal_cov(W.T @ W, sigma, dev_homo.shape[0])
        llh += Gauss(cov=cov).log_p(dev_homo.ravel())
    return llh


def _construct_marginal_cov(heter_cov, noise_cov, n_obs):
    cov = np.tile(heter_cov, (n_obs, n_obs))
    rr, cc = noise_cov.shape
    r, c = 0, 0
    for _ in range(n_obs):
        cov[r:r+rr, c:c+cc] += noise_cov
        r += rr
        c += cc
    return cov
