import numpy as np
# import cupy as np
from numpy.linalg import svd
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from scipy.linalg import cholesky
from numpy.linalg import inv
from numpy.linalg import slogdet
from utili_ import h52dict
from utili_ import dict2h5

class SImPLDA:
    """A class for  snr-indendpent mixture of PLDA
    """

    def __init__(self):

        self.varphi = []
        self.m = []
        self.V = []
        self.Sigma = []
        self.const = []
        self.logDetCov = []
        self.Icov = []
        self.logDetCov2 = []
        self.Icov2 = []
        self.P = []
        self.Q = []
        self.projmat1 = []
        self.projmat2 = []
        self.meanVec1 = []

    def init_mPLDA(self, X, IDmat, M, K):
        """
         function [varphi, mu, V, Sigma] = init_mPLDA(X, IDmat, M, K)
         Initialize the mPLDA model parameters
         Input:
           X     - D x H matrix containing i-vectors of the same speaker in columns
           IDmat - ID matrix containing speaker information (N_DATA x N_INDIV)
           M     - Dim of latent vectors
           K     - No. of mixtures
         Output:
           varphi - Mixture weights
           m      - Cell array containing nMix (D x 1) mean vectors
           V      - Cell array containing nMix (D x M) factor loading matrices
           Sigma  - Cell array containing nMix (D x D) covariance matrix of noise e
        """

        D = X.shape[0]
        T = X.shape[1]
        Data = X - np.tile(np.mean(X, axis=1).reshape(-1, 1), [1, T])

        clusterMeans = np.dot(np.dot(Data, IDmat), np.diag(1 / sum(IDmat)))
        evec, _ = self.trainPCA(clusterMeans)
        V = np.empty(shape=(1, K), dtype=np.object)
        for k in range(K):
            V[0, k] = evec[:, :M]

        clusterMeans = np.dot(np.dot(X, IDmat), np.diag(1 / sum(IDmat)))
        clf_ = KMeans(n_clusters=K)
        clf_.fit(clusterMeans.T)
        cmeans = clf_.cluster_centers_

        m = np.empty(shape=(1, K), dtype=object)
        for k in range(K):
            m[0, k] = cmeans[k, :]

        Sigma = np.empty(shape=(1, K), dtype=object)
        for k in range(K):
            Sigma[0, k] = 0.01 * np.eye(D)

        varphi = np.ones(shape=(K, 1)) / K

        return varphi, m, V, Sigma

    def trainPCA(self, data):
        """
         function  [princComp meanVec] = trainPCA(data)

         Principla Component Analysis (PCA)

           Input:
               Data    - NFeature  x NSample   Training data

           Output:
               princComp   - Principal Expectation of latent varialbe h and latent noise
               meanVec     - Mean vector of the data
        """

        nData = data.shape[1]
        meanVec = np.mean(data, axis=1).reshape(-1, 1)
        data = data - np.tile(meanVec, [1, nData])
        XXT = np.dot(data.T, data)

        _, LSq, V = svd(XXT)
        V = V.T

        LInv = 1. / np.sqrt(LSq)
        princComp = np.dot(np.dot(data, V), np.diag(LInv))
        return princComp, meanVec

    def get_alignment(self, X, sessions):
        """
         function [H_set, postp] = get_alignment(X, sessions, varphi, m, V, Sigma)

         Find the mixture alignment of each utterance

           Input:
               X           - DxT matrix containing T i-vectors in columns
               sessions    - Cell array containing the session indexes (index to L) of N speakers
               varphi      - Kx1 vector containing mixture weights
               m           - Kx1 cell array containing mean i-vectors
               V           - Kx1 cell array containing DxM loading matrices
               Sigma       - Kx1 cell array containing DxD covariance matrices of residue
           Output:
               H_set       - NxK cell matrix, where H_set{i,k} is an array containing mixture indexes
                             of H_i alignments
               postp       - Nx1 cell array, where postp{i} is a H_i x K matrix containing the posterior
                             probability of K mixtures for H_i sessions.
        """

        N = len(sessions)
        K = len(self.varphi)
        H_set = np.empty(shape=(N, K), dtype=object)
        postp = np.empty(shape=(N, 1), dtype=object)
        for i in range(N):  # For each spk
            ss = sessions[i, 0][0]  # Sessions of spk i
            postp[i, 0] = self.posterior(X[:, ss])  # <y_ikj|X> where j=1,...,H_i
            maxk = np.argmax(postp[i, 0], axis=1)
            for k in range(K):
                H_set[i, k] = np.where(maxk == k)
        return H_set, postp

    def posterior(self, X):
        """
         function postx = posterior(X, varphi, m, V, Sigma)

         Compute the posterior probababilities of vectors X in columns

           Input:
               X           - DxH_i matrix containing H_i i-vectors in columns
               varphi      - Kx1 vector containing mixture weights
               m           - Kx1 cell array containing mean i-vectors
               V           - Kx1 cell array containing DxM loading matrices
               Sigma       - Kx1 cell array containing DxD covariance matrices of residue
           Output:
               postp       - H_i x K matrix containing the posterior
                             probability of K mixtures for H_i sessions.
        """

        K = len(self.varphi)
        Hi = X.shape[1]
        likeh = np.zeros(shape=(Hi, K), dtype=np.float64)
        for k in range(K):
            likeh[:, k] = self.varphi[k] * multivariate_normal.pdf(X.T, mean=self.m[0, k].flatten(), cov=np.dot(self.V[0, k], self.V[0, k].T) + self.Sigma[0, k])
        post = np.zeros(shape=(Hi, K), dtype=np.float64)
        for j in range(Hi):
            post[j, :] = likeh[j, :] / np.sum(likeh[j, :])
        return post

    def update_mixture(self, X, sessions, posty):
        """
         function [varphi,m] = update_mixture(X,sessions,posty)

         Update mixture parameters: varphi and m

           Input:
               X           - D x T data matrix with D-dim column vectors
               sessions    - Cell array containing the session indexes (index to L) of N speakers
               posty       - Nx1 cell array, where postp{i} is a H_i x K matrix containing the posterior
                             probability of K mixtures for H_i sessions.
           Output:
               varphi      - Kx1 vector containing mixture weights
               m           - Kx1 cell array containing mean i-vectors
        """

        K = posty[0, 0].shape[1]
        sum_Ey = np.zeros(shape=(K, 1), dtype=np.float64)
        D = X.shape[0]
        N = len(sessions)
        m = np.empty(shape=(1, K), dtype=object)
        for k in range(K):
            sum3 = np.zeros(shape=(D, 1), dtype=np.float64)
            sum4 = 0
            for i in range(N):
                ss = sessions[i, 0][0]
                for j in range(len(ss)):
                    E_yijk = posty[i, 0][j, k]
                    sum3 = sum3 + E_yijk * (X[:, ss[j]].reshape(-1, 1))
                    sum4 = sum4 + E_yijk
            m[0, k] = sum3 / sum4
            sum_Ey[k] = sum4
        varphi = sum_Ey / np.sum(sum_Ey)
        return varphi, m

    def update_Sigma(self, X, E_y, E_z, sessions):
        """
         function Sigma = update_Sigma(X, m, V, E_y, E_z, sessions)

         Update non-tied covariance matrices, i.e., Sigma{k} are not identical

           Input:
               X           - D x T data matrix with D-dim column vectors
               sessions    - Cell array containing the session indexes (index to L) of N speakers
               m           - Kx1 cell array containing the mean i-vectors of mixtures
               V           - Cell array of loading matrices
               E_y         - Cell array of H_i dim vectors E_y{i,k}(j) = <y_ijk|L>
               E_z         - Cell array of M-dim vector: E_z{i} = <z_i|y_i..,X,L>
           Output:
               Sigma       - Cell array of covariance matrices
        """

        K = len(self.m[0])
        Sigma = np.empty(shape=(1, K), dtype=object)
        N = len(sessions)
        D = X.shape[0]
        for k in range(K):
            sum1 = np.zeros(shape=(D, D), dtype=np.float64)
            sum2 = 0
            for i in range(N):
                ss = sessions[i, 0][0]
                for j in range(len(ss)):
                    x = X[:, ss[j]]
                    E_yijk = E_y[i, k][j]    # E_y[i, k][j]
                    temp = x.reshape(-1, 1) - self.m[0, k]
                    sum1 = sum1 + E_yijk * np.dot(temp, temp.T) - E_yijk * np.dot(np.dot(self.V[0, k], E_z[0, i]), temp.T)
                    sum2 = sum2 + E_yijk
            Sigma[0, k] = sum1 / sum2
        return Sigma

    def auxFunc(self, X, E_y, E_z, E_zz, sessions):
        """
         function  Q = auxFunc(varphi,m,V,Sigma, X, E_y, E_z, E_zz, sessions)

           Input:
             pi        - Mixture weights
             mu        - K x 1 vector containing the mean of utt length
             sigma     - K x 1 vector containing the stddev of utt length
             m         - Cell array containing nMix (D x 1) mean vectors
             V         - Cell array containing K (D x M) factor loading matrices
             Sigma     - (D x D) full covariance matrix of noise e
             X         - Training data (NFeature  x NSample)
             L         - Length of utterance, 1 x NSample
             E_y       - E_y{i,k}(j) = <y_ijk|L>
             E_z       - E_z{i} = <z_i|y_i..|X,L>
             E_zz      - E_zz{i} = <z_i z_i'|y_i..,X,L>
             sessions  - Cell array containing no. of sessions for each speaker

           Output:
               Q       - complete data log-likelihood
        """

        N = len(sessions)  # No. of speakers
        K = len(self.varphi)
        logDetSig = np.zeros(shape=(1, K), dtype=np.float64)
        ISig = np.empty(shape=(1, K), dtype=object)
        for k in range(K):
            logDetSig[0, k] = 2 * np.sum(np.log(np.diag(cholesky(self.Sigma[0, k]))))  # log|Sigma|
            ISig[0, k] = inv(self.Sigma[0, k])
        sum2 = np.zeros(shape=(1, K), dtype=np.float64)
        sum3 = np.zeros(shape=(1, K), dtype=np.float64)
        sum4 = np.zeros(shape=(1, K), dtype=np.float64)
        sum_ij = np.zeros(shape=(1, K), dtype=np.float64)
        for i in range(N):
            ss = sessions[i, 0][0]
            for k in range(K):
                for j in range(len(ss)):
                    x = X[:, ss[j]]
                    temp = x.reshape(-1, 1) - self.m[0, k]
                    E_yijk = E_y[i, k][j]
                    sum2[0, k] = sum2[0, k] + E_yijk * (np.log(self.varphi[k]) - 0.5 * logDetSig[0, k] - 0.5 * np.dot(temp.T, np.dot(ISig[0, k], temp)))
                    sum3[0, k] = sum3[0, k] + np.dot((np.dot(temp.T, ISig[0, k])), (E_yijk * np.dot(self.V[0, k], E_z[0, i])))
                    sum4[0, k] = sum4[0, k] - 0.5 * np.trace(np.dot((np.dot(self.V[0, k].T, np.dot(ISig[0, k], self.V[0, k])) + 1), E_yijk * E_zz[0, i]))
                    sum_ij[0, k] = sum_ij[0, k] + E_yijk
        S = sum2 + sum3 + sum4
        print('Likelihood:')
        for k in range(K):
            print('(%.2f, %.2f)' % (S[0, k] / X.shape[1], sum_ij[0, k] / X.shape[1]), end='')
        print('\n')
        Q = np.sum(S)
        return Q

    def prepare_scoring(self, alpha):
        """
         function [P,Q,const, Icov, logDetCov, Icov2, logDetCov2] = prepare_scoring(m, V, Sigma, alpha)
         Precompute matrices for fast scoring
           Input:
               m           - Cell array containing means of i-vectors
               V           - Cell array containing K DxM Speaker factor loading matrix
               Sigma       - Cell array containing K DxD full residual cov matrices
               alpha       - Scaling factor for computing logDetCov, i.e., |alpha*C|,
                             to avoid overflow and underflow
           Output:
               const       - Const. term of log-likelihood
               P,Q         - Cell array containing K matrices
               Icov        - Cell array containing Inverse of V{k}*V{k}'+Sigma{k}
               logDetCov   - Cell array containing logDet of alpha*cov matrices
               Icov2       - Same as Icov but based on the stacking of two V
               logDetCov2  - Same as logDetCov but based on the stacking of two V
         Bug fixed (Jun 15):
               Sigma2 should be computed inside ks and kt double-loop
               Original code:
                 for ks=1:K,
                     Icov{ks} = inv(V{ks}*V{ks}' + Sigma{ks});
                     logDetCov{ks} = logDet(alpha*(V{ks}*V{ks}' + Sigma{ks}));
                     Sigma2{ks} = [Sigma{ks} zeros(D,D); zeros(D,D) Sigma{ks}]; <--- Bug here, see below fix
                     for kt=1:K,
                         V2 = [V{ks}; V{kt}];
                         Icov2{ks,kt} = inv(V2*V2' + Sigma2{ks});
                         logDetCov2{ks,kt} = logDet(alpha*(V2*V2' + Sigma2{ks}));
                     end
                 end
                Effect of fixing this bug: Increase EER but decrease minDCF
        """

        D = self.Sigma[0, 0].shape[0]
        K = len(self.m[0])
        logDetCov = np.empty(shape=(1, K), dtype=object)
        Icov = np.empty(shape=(1, K), dtype=object)
        logDetCov2 = np.empty(shape=(K, K), dtype=object)
        Icov2 = np.empty(shape=(K, K), dtype=object)
        for ks in range(K):
            Icov[0, ks] = inv(np.dot(self.V[0, ks], self.V[0, ks].T) + self.Sigma[0, ks])
            logDetCov[0, ks] = slogdet((alpha * (np.dot(self.V[0, ks], self.V[0, ks].T) + self.Sigma[0, ks])))[1]
            for kt in range(K):
                V2 = np.vstack((self.V[0, ks], self.V[0, kt]))
                Sigma2 = np.vstack((np.hstack((self.Sigma[0, ks], np.zeros(shape=(D, D), dtype=np.float64))),  \
                                    np.hstack((np.zeros(shape=(D, D), dtype=np.float64), self.Sigma[0, kt]))))
                Icov2[ks, kt] = inv(np.dot(V2, V2.T) + Sigma2)
                logDetCov2[ks, kt] = slogdet((alpha * (np.dot(V2, V2.T) + Sigma2)))[1]

        # Compute matrices P and Q  and log-likelihood constant for each mixture
        P = np.empty(shape=(K, K), dtype=object)
        Q = np.empty(shape=(K, K), dtype=object)
        Phi = np.empty(shape=(1, K), dtype=object)
        Psi = np.empty(shape=(K, K), dtype=object)
        const = np.zeros(shape=(K, K), dtype=np.float64)
        for i in range(K):
            Phi[0, i] = np.dot(self.V[0, i], self.V[0, i].T) + self.Sigma[0, i]
            for j in range(K):
                Psi[i, j] = np.dot(self.V[0, i], self.V[0, j].T)
        for ks in range(K):
            for kt in range(K):
                Q[ks, kt] = inv(Phi[0, ks]) - inv(Phi[0, ks] - np.dot(np.dot(Psi[ks, kt], inv(Phi[0, kt])), Psi[kt, ks]))
                P[ks, kt] = np.dot(np.dot(inv(Phi[0, ks]), Psi[ks, kt]), inv(Phi[0, kt] - np.dot(np.dot(Psi[kt, ks], inv(Phi[0, ks])), Psi[ks, kt])))
        for ks in range(K):
            for kt in range(K):
                Sig_0 = np.zeros(shape=Phi[0, ks].shape, dtype=np.float64)
                D1 = np.vstack((np.hstack((Phi[0, ks], Psi[ks, kt])), np.hstack((Psi[kt, ks], Phi[0, kt]))))
                D2 = np.vstack((np.hstack((Phi[0, ks], Sig_0)), np.hstack((Sig_0, Phi[0, kt]))))
                const[ks, kt] = (-0.5) * slogdet(D1)[1] + 0.5 * slogdet(D2)[1] \
                                - np.dot(np.dot(self.m[0, ks].T, Q[ks, kt]), self.m[0, ks]) - np.dot(np.dot(self.m[0, kt].T, Q[kt, ks]), self.m[0, kt]) \
                                - np.dot(np.dot(self.m[0, ks].T, P[ks, kt]), self.m[0, kt]) - np.dot(np.dot(self.m[0, kt].T, P[ks, kt].T), self.m[0, ks])

        return P, Q, const, Icov, logDetCov, Icov2, logDetCov2

    def perform_EM(self, X, IDmat, D, N, M, K, nIters):

        # Prepare EM: Create storage for posterior expectations
        # To reduce memory consumption, we do not store E_yz and E_yzz. These
        # posterior expectation should be computed on the fly.
        # Instead, we store E_z{i} and E_zz{i}. They are independent of j and k
        E_y = np.empty(shape=(N, K), dtype=object)
        E_z = np.empty(shape=(1, N), dtype=object)
        E_zz = np.empty(shape=(1, N), dtype=object)
        H = np.empty(shape=(N, 1), dtype=object)
        sessions = np.empty(shape=(N, 1), dtype=object)

        # Find the sessions of each training speaker
        for i in range(N):
            sessions[i, 0] = np.where(IDmat[:, i] == 1)
            H[i, 0] = len(sessions[i, 0][0])

        # Start mPLDA EM
        for t in range(nIters):

            # Stage 1: Optimize mixture parameters varphi{k}, m{k}, V{k}, and Sigma{k}
            # E-step: Compute posterior expectation
            _, posty = self.get_alignment(X, sessions)  # Find the mixture alignment of each utterance  checked
            self.varphi, self.m = self.update_mixture(X, sessions, posty)  # checked

            # Stage 2: Use updated mixture parameters to perform EM to optimize V{k} and Sigma{k}
            # E-step: Compute posterior expectation
            _, posty = self.get_alignment(X, sessions)  # Find the mixture alignment of each utterance
            for i in range(N):  # For each spk
                ss = sessions[i, 0][0]  # Sessions of spk i
                sum_jk = np.zeros(shape=(M, M), dtype=np.float64)
                for k in range(K):
                    E_y[i, k] = posty[i, 0][:, k]
                    for j in range(H[i, 0]):
                        sum_jk = sum_jk + posty[i, 0][j, k] * np.dot(np.dot(self.V[0, k].T, inv(self.Sigma[0, k])), self.V[0, k])
                IL = np.dot(np.eye(M), inv(np.eye(M) + sum_jk))
                sum_kj = np.zeros(shape=(M, 1), dtype=np.float64)
                for k in range(K):
                    for j in range(H[i, 0]):
                        x = X[:, ss[j]]
                        sum_kj = sum_kj + posty[i, 0][j, k] * np.dot(np.dot(self.V[0, k].T, inv(self.Sigma[0, k])), x.reshape(-1, 1) - self.m[0, k])
                E_z[0, i] = np.dot(IL, sum_kj)  # <z_i|y_i..,X>
                E_zz[0, i] = IL + np.dot(E_z[0, i], E_z[0, i].T)  # <z_i|y_i..,X><z_i|y_i..,X>'

            # M - step
            # Update V_k
            for k in range(K):
                sum1 = np.zeros(shape=(D, M), dtype=np.float64)
                sum2 = np.zeros(shape=(M, M), dtype=np.float64)
                for i in range(N):
                    ss = sessions[i, 0][0]
                    for j in range(len(ss)):
                        x = X[:, ss[j]]
                        E_yijk = E_y[i, k][j]
                        sum1 = sum1 + E_yijk * np.dot(x.reshape(-1, 1) - self.m[0, k], E_z[0, i].T)
                        sum2 = sum2 + E_yijk * E_zz[0, i]
                self.V[0, k] = np.dot(sum1, inv(sum2))

            # Update Sigma_k
            self.Sigma = self.update_Sigma(X, E_y, E_z, sessions)  # checked

            # Compute Q-func
            Qnew = self.auxFunc(X, E_y, E_z, E_zz, sessions)
            if t > 1:
                perinc = abs(100 * (Qnew - Qold) / Qold)
                if Qnew > Qold:
                    print('Iteration %d: llh = %.2f; %%increase = %.2f%%\n' % (t, Qnew, perinc))
                else:
                    print('Iteration %d: llh = %.2f; %%decrease = %.2f%%\n' % (t, Qnew, perinc))
            Qold = Qnew

    def save_model(self, GPLDA_file):
        np.save(GPLDA_file, self.__dict__)

    def load_model(self, GPLDA_file):
        temp_dict = np.load(GPLDA_file, allow_pickle=True).reshape(1)[0]
        for key in self.__dict__:
            setattr(self, key, temp_dict[key])


def mPLDA_train5(X, L, IDmat, nIters, M, K):
    """
     function mplda = mPLDA_train5(X, L, IDmat, nIters, nSpkFacs, nMix)
     Implement mixture of PLDA training based on Hinton's mixture of FA
     Ghahramani, Zoubin, and Geoffrey E. Hinton. The EM algorithm for mixtures of factor analyzers.
     Vol. 60. Technical Report CRG-TR-96-1, University of Toronto, 1996.
     However, unlike Hinton's paper, this algorithm sum over all mixture k when computing
     the posterior covariance of the hidden factors.

     This function works with mPLDA_GroupScoring5.m

     FA model: p(x) = sum_k pi_k * Gauss(mu_k, V_k*V_k' + Sigma)   (Eq. 7 of the above paper)
     where pi_k, mu_k and V_k are the mixture, mean and loading matrix of kth FA,
     and Sigma is the noise (residue) covariance

     Input:
       X         - D x T data matrix with D-dim column vectors
       L         - A dummy array for compatibility with other mPLDA_trainX.m function
       IDmat     - ID matrix (numVecs x numSpks). For each row, the column with a 1
                   means that the i-vector is produced by the corresponding speaker.
                   The rest should be zero.
       nSpkfacs  - No. of speaker factors (latent variables)
       nMix      - No. of factor analyzers
       nIters    - No. of EM iterations
     Output:
       mplda     - A structure containing the following fields
          varphi    - Mixture weights
          mu        - K x 1 vector containing the mean of utt length
          sigma     - K x 1 vector containing the stddev of utt length
          m         - Cell array containing K cells, each contains a (D x 1) mean vectors
          V         - Cell array containing K cells, each contains a (D x M) factor loading matrices
          Sigma     - Cell array containing K cells, each contains a (D x D) full covariance matrix of noise e
          const     - const for computing log-likelihood during scoring (for future use)
          logDetCov - Output of private function prepare_scoring() for fast scoring
          Icov      - Output of private function prepare_scoring() for fast scoring
          logDetCov2- Output of private function prepare_scoring() for fast scoring
          Ico2      - Output of private function prepare_scoring() for fast scoring
          P,Q       - Cell array of P and Q matrices for fast scoring (for future use)

     Example useage:
        See comp_mGPLDA.m
     By M.W. Mak
    """

    D = X.shape[0]          # Dim of i-vectors
    N = IDmat.shape[1]      # No. of speakers

    # Initialize mPLDA model parameters
    mPLDA = SImPLDA()
    mPLDA.varphi, mPLDA.m, mPLDA.V, mPLDA.Sigma = mPLDA.init_mPLDA(X, IDmat, M, K)

    # train mPLDA model
    # nIters =1
    mPLDA.perform_EM(X, IDmat, D, N, M, K, nIters)

    # Prepare paras for speeding up scoring
    mPLDA.P, mPLDA.Q, mPLDA.const, mPLDA.Icov, mPLDA.logDetCov, mPLDA.Icov2, mPLDA.logDetCov2 = mPLDA.prepare_scoring(5)  # checked

    return mPLDA
