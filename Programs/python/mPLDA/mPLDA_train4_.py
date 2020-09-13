import numpy as np
from numpy.linalg import svd
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal
from numpy.linalg import inv
from scipy.linalg import cholesky
from numpy.linalg import slogdet

def show_mixture_stat(pi, mu, sigma):
    """
    Display the mixture statistics
    """

    for k in range(len(pi)):
        print('(%.1f,%.1f,%.1f) ' % (pi[k, 0], mu[k, 0], sigma[k, 0]), end='')
    print('\n')


class SDmPLDA:
    """A class for snr-dependent mixture of Gaussian PLDA
    """

    def __init__(self):

        self.pi = []
        self.mu = []
        self.sigma = []
        self.m =[]
        self.V =[]
        self.Sigma =[]
        self.const = []
        self.logDetCov = []
        self.Icov = []
        self.logDetCov2 = []
        self.Icov2 = []
        self.P =[]
        self.Q =[]
        self.projmat1 = []
        self.projmat2 = []
        self.meanVec1 = []

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

    def init_mPLDA(self, X, L, IDmat, M, K):
        """
         function mplda = init_mPLDA(X, L, IDmat, M, K)
         Initialize the mPLDA model parameters
         Input:
           X     - D x H matrix containing i-vectors of the same speaker in columns
           L     - 1 x H vector containing length of utterances
           IDmat - ID matrix containing speaker information (N_DATA x N_INDIV)
           M     - Dim of latent vectors
           K     - No. of mixtures
         Output:
               pi    - Mixture weights
               mu    - K x 1 vector containing the mean of utt length
               sigma - K x 1 vector containing the stddev of utt length
               m     - Cell array containing K (D x 1) mean vectors
               V     - Cell array of (D x M) factor loading matrices
               Sigma - (D x D) full covariance matrix of noise e
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

        pi = np.zeros(shape=(K, 1))
        mu = np.zeros(shape=(K, 1))
        sigma = np.zeros(shape=(K, 1))

        clf_ = KMeans(n_clusters=K)
        clf_.fit(L.reshape(-1, 1))
        cidx = clf_.labels_

        for k in range(K):
            pi[k] = len(np.where(cidx == k)[0]) / len(cidx)
            mu[k] = np.mean(L[np.where(cidx == k)[0]])
            sigma[k] = np.std(L[np.where(cidx == k)[0]])

        return pi, mu, sigma, m, V, Sigma

    def posterior_y(self, l):
        """
         function Ey = posterior_y(l, pi, mu, sigma)
         Return a matrix of size length(l) x K the posterior prob. of l.
         Input:
           l     - H-dim vector containing length of utterances of spk i
           pi    - K-dim vector containing the mixture weights
           mu    - K-dim vector containing the means of utt length of speaker i
           sigma - K-dim vector containing the stddev of utt length of speaker i
         Output:
           posty - H x K matrix containing posterior p(y_ijk|X), j=1..H
        """

        K = len(self.pi)
        H = len(l)
        posty = np.ones(shape=(H, K), dtype=np.float64)
        for j in range(H):
            wlh = np.zeros(shape=(1, K), dtype=np.float64)
            for r in range(K):
                wlh[0, r] = self.pi[r, 0] * multivariate_normal.pdf(l[j], mean=self.mu[r, 0].flatten(), cov = np.square(self.sigma[r, 0]))
            sum_wlh = np.sum(wlh)
            assert np.isnan(sum_wlh).any() == 0, 'Assertion in posteror_y_.py: sum(wlh) is NaN'
            for k in range(K):
                if sum_wlh > 0:
                    posty[j, k] = wlh[0, k] / sum_wlh
        return posty

    def get_alignment(self, L, sessions):
        """
         function [H_set, postp] = get_alignment(L, sessions, pi, mu, sigma)

         Find the mixture alignment of each utterance

           Input:
               L           - Array containing SNR or utterance length of T utterances
               sessions    - Cell array containing the session indexes (index to L) of N speakers
               pi,mu,sigma - 1-D GMM of length or SNR model
           Output:
               H_set       - NxK cell matrix, where H_set{i,k} is an array containing mixture indexes
                             of H_i alignments
               postp       - Nx1 cell array, where postp{i} is a H_i x K matrix containing the posterior
                             probability of K mixtures for H_i sessions.
        """

        N = len(sessions)
        K = len(self.pi)
        H_set = np.empty(shape=(N, K), dtype=object)
        postp = np.empty(shape=(N, 1), dtype=object)
        for i in range(N):  # For each spk
            ss = sessions[i, 0][0]  # Sessions of spk i
            postp[i, 0] = self.posterior_y(L[ss])      # <y_ikj|L> where j=1,...,H_
            maxk = np.argmax(postp[i, 0], axis=1)
            for k in range(K):
                H_set[i, k] = np.where(maxk == k)
        return H_set, postp

    def update_mixture(self, L, X, sessions, posty):
        """
         function [pi,mu,sigma,m] = update_mixture(pi,mu,sigma,m,L,X,sessions,posty)

         Update mixture parameters: pi, mu, sigma, and m

           Input:
               X           - D x T data matrix with D-dim column vectors
               L           - Array containing SNR or utterance length of T utterances
               sessions    - Cell array containing the session indexes (index to L) of N speakers
               pi,mu,sigma - 1-D GMM of length or SNR model
               m           - Kx1 cell array containing the mean i-vectors of mixtures
               posty       - Nx1 cell array, where postp{i} is a H_i x K matrix containing the posterior
                             probability of K mixtures for H_i sessions.
           Output:
               pi,mu,sigma,m
        """

        K = len(self.pi)
        sum_Ey = np.zeros(shape=(K, 1), dtype=np.float64)
        D = X.shape[0]
        N = len(sessions)
        for k in range(K):
            sum1 = 0
            sum2 = 0
            sum3 = np.zeros(shape=(D, 1), dtype=np.float64)
            sum4 = 0
            for i in range(N):
                ss = sessions[i, 0][0]
                for j in range(len(ss)):
                    E_yijk = posty[i, 0][j, k]
                    sum1 = sum1 + E_yijk * L[ss[j]]
                    sum2 = sum2 + E_yijk * ((L[ss[j]] - self.mu[k, 0]) * (L[ss[j]] - self.mu[k, 0]))
                    sum3 = sum3 + E_yijk * (X[:, ss[j]].reshape(-1, 1))
                    sum4 = sum4 + E_yijk
            self.mu[k, 0] = sum1 / sum4
            self.sigma[k, 0] = np.sqrt(sum2 / sum4)
            self.m[0, k] = sum3 / sum4
            sum_Ey[k, 0] = sum4
        pi = sum_Ey / np.sum(sum_Ey)
        return pi, self.mu, self.sigma, self.m

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
                    E_yijk = E_y[i, k][j]  # E_y[i, k][j]
                    temp = x.reshape(-1, 1) - self.m[0, k]
                    sum1 = sum1 + E_yijk * np.dot(temp, temp.T) - E_yijk * np.dot(np.dot(self.V[0, k], E_z[0, i]), temp.T)
                    sum2 = sum2 + E_yijk
            Sigma[0, k] = sum1 / sum2
        return Sigma

    def auxFunc(self, X, L, E_y, E_z, E_zz, sessions):
        """
         function  Q = auxFunc(pi,mu,sigma,m,V,Sigma, X, L, E_y, E_z, E_zz,, sessions);

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
             E_z       - E_z{i} = <z_i|y_i..,X,L>
             E_zz      - Cell array of MxM matrix E_zz{i} = <z_i z_i|y_i..,X>
             sessions  - Cell array containing no. of sessions for each speaker

           Output:
               Q       - complete data log-likelihood
        """

        N = len(sessions)  # No. of speakers
        K = len(self.pi)
        logDetSig = np.zeros(shape=(1, K), dtype=np.float64)
        ISig = np.empty(shape=(1, K), dtype=object)
        for k in range(K):
            logDetSig[0, k] = 2 * np.sum(np.log(np.diag(cholesky(self.Sigma[0, k]))))  # log|Sigma|
            ISig[0, k] = inv(self.Sigma[0, k])
        sum1 = np.zeros(shape=(1, K), dtype=np.float64)
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
                    sum1[0, k] = sum1[0, k] + E_yijk * ((-1) * np.log(self.sigma[k, 0]) - (0.5 / (np.square(self.sigma[k, 0]))) * np.square(L[ss[j]] - self.mu[k, 0]))
                    sum2[0, k] = sum2[0, k] + E_yijk * (np.log(self.pi[k, 0]) - 0.5 * logDetSig[0, k] - 0.5 * np.dot(temp.T, np.dot(ISig[0, k], temp)))
                    sum3[0, k] = sum3[0, k] + np.dot((np.dot(temp.T, ISig[0, k])), (E_yijk * np.dot(self.V[0, k], E_z[0, i])))
                    sum4[0, k] = sum4[0, k] - 0.5 * np.trace(np.dot((np.dot(self.V[0, k].T, np.dot(ISig[0, k], self.V[0, k])) + 1), E_yijk * E_zz[0, i]))
                    sum_ij[0, k] = sum_ij[0, k] + E_yijk
        S = sum1 + sum2 + sum3 + sum4
        print('Likelihood:')
        for k in range(K):
            print('(%.2f, %.2f)' % (S[0, k] / X.shape[1], sum_ij[0, k] / X.shape[1]), end='')
        print('\n')
        Q = np.sum(S)
        return Q

    def perform_EM(self, X, L, IDmat, nIters, D, N, M, K):

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

             # Stage 1: Optimize mixture parameters pi{k},sigma{k}, mu{k}, and m{k}
            _, posty = self.get_alignment(L, sessions)  # Find the mixture alignment of each utterance
            self.pi, self.mu, self.sigma, self.m = self.update_mixture(L, X, sessions, posty)

             # Stage 2: Use updated mixture parameters to perform EM to optimize V{k} and Sigma{k}
             # E-step: Compute posterior expectation
            _, posty = self.get_alignment(L, sessions)   # Find the mixture alignment of each utterance
            for i in range(N):  # For each spk
                ss = sessions[i, 0][0]  # Sessions of spk i
                sum_k = np.zeros(shape=(M, M), dtype=np.float64)
                for k in range(K):
                    E_y[i, k] = posty[i, 0][:, k]
                    for j in range(H[i, 0]):
                        sum_k = sum_k + posty[i, 0][j, k] * np.dot(np.dot(self.V[0, k].T, inv(self.Sigma[0, k])), self.V[0, k])
                IL = np.dot(np.eye(M), inv(np.eye(M) + sum_k))
                sum_kj = np.zeros(shape=(M, 1), dtype=np.float64)
                for k in range(K):
                    for j in range(H[i, 0]):
                        x = X[:, ss[j]]
                        sum_kj = sum_kj + posty[i, 0][j, k] * np.dot(np.dot(self.V[0, k].T, inv(self.Sigma[0, k])), x.reshape(-1, 1) - self.m[0, k])
                E_z[0, i] = np.dot(IL, sum_kj)  # <z_i|y_i..,X>
                E_zz[0, i] = IL + np.dot(E_z[0, i], E_z[0, i].T)  # <z_i|y_i..,X><z_i|y_i..,X>'

            #  M-step
            #  Update V_k
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

            #  Update tied Sigma_k, equal for all k
            self.Sigma = self.update_Sigma(X, E_y, E_z, sessions)

            # Show some statistics about <y_ijk|L>
            show_mixture_stat(self.pi, self.mu, self.sigma)

            # Compute Q - func
            Qnew = self.auxFunc(X, L, E_y, E_z, E_zz, sessions)
            if t > 1:
                perinc = abs(100 * (Qnew - Qold) / Qold)
                if Qnew > Qold:
                    print('Iteration %d: llh = %.2f; %%increase = %.2f%%\n' % (t, Qnew, perinc))
                else:
                    print('Iteration %d: llh = %.2f; %%decrease = %.2f%%\n' % (t, Qnew, perinc))
            Qold = Qnew


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
               Icov        - Cell array containing Inverse of V*V'+Sigma{k}
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
                                - 0.5 * np.dot(np.dot(self.m[0, ks].T, Q[ks, kt]), self.m[0, ks]) - 0.5 * np.dot(np.dot(self.m[0, kt].T, Q[kt, ks]), self.m[0, kt]) \
                                - 0.5 * np.dot(np.dot(self.m[0, ks].T, P[ks, kt]), self.m[0, kt]) - 0.5 * np.dot(np.dot(self.m[0, kt].T, P[ks, kt].T), self.m[0, ks])

        return P, Q, const, Icov, logDetCov, Icov2, logDetCov2

    def save_model(self, GPLDA_file):
        np.save(GPLDA_file, self.__dict__)

    def load_model(self, GPLDA_file):
        temp_dict = np.load(GPLDA_file, allow_pickle=True).reshape(1)[0]
        for key in self.__dict__:
            setattr(self, key, temp_dict[key])

def mPLDA_train4(X, L, IDmat, nIters, M, K):
    """
     function mplda = mPLDA_train4(X, L, IDmat, nIters, M, K)
     Implement snr-dependent mixture of Gaussian PLDA (training).
     The mPLDA model has K loading matrices and K covmat.
     This function assumes that each speaker share the same z_i and that z_i depends
     on y_ijk for all j in H_ik and for all k. These assumptions lead to the sum over
     all i-vecs aligned to mixture k when computing <z_i|y_i..,X>. The full set of
     equations are given in Eq. 39 and 40 of the supplmentary materials.
     This function uses the 2-stage EM as described in Tipping and Bishop's paper.

     Input:
       X         - D x T data matrix with D-dim column vectors
       L         - T-dim array containing the SNR or the number of frames in the utts
       IDmat     - ID matrix (numVecs x numSpks). For each row, the column with a 1
                   means that the i-vector is produced by the corresponding speaker.
                   The rest should be zero.
       M         - No. of speaker factors (latent variables)
       K         - No. of factor analyzers
       nIters    - No. of EM iterations

     Output:
       mplda     - A structure containing the following fields
          pi        - Mixture weights
          mu        - K x 1 vector containing the mean of SNR
          sigma     - K x 1 vector containing the stddev of utt length
          m         - Cell array containing K cells, each contains a (D x 1) mean vectors
          V         - Cell array containing K cells, each contains a (D x M) factor loading matrices
          Sigma     - Cell array containing K cells, each contains a (D x D) full covariance matrix of noise e
          const     - const for computing log-likelihood during scoring
          logDetCov - Output of private function prepare_scoring() for fast scoring
          Icov      - Output of private function prepare_scoring() for fast scoring
          logDetCov2- Output of private function prepare_scoring() for fast scoring
          Ico2      - Output of private function prepare_scoring() for fast scoring
          P,Q       - Cell array of P and Q matrices for fast scoring (used by my APSIPA'15 paper)

     Bug fix:
       2/8/15       - sum_ij V{k}<y_ijk z_ik|X> is not zero. So, it is necessary to subtract it
                      from sum_ij <y_ijk|X>x_ij when computing m{k}. However, this fix leads to
                      poor performance. According to Tipping and Bishop paper "Mixtures of
                      Probabilistic Principal Component Analysers, if we use the so-called 2-step EM,
                      we do not need to subtract V{k}<y_ijk z_ik|X> from sum_ij <y_ijk|X>x_ij, i.e.,
                      m{k} = sum3/sum4;

     By M.W. Mak, July 2015
    """

    D = X.shape[0]  # Dim of i-vectors
    N = IDmat.shape[1]  # No. of speakers

    # Initialize mPLDA model parameters
    mPLDA = SDmPLDA()
    mPLDA.pi, mPLDA.mu, mPLDA.sigma, mPLDA.m, mPLDA.V, mPLDA.Sigma = mPLDA.init_mPLDA(X, L, IDmat, M, K)

    # train mPLDA model
    mPLDA.perform_EM(X, L, IDmat, nIters, D, N, M, K)

    # Prepare paras for speeding up scoring
    mPLDA.P, mPLDA.Q, mPLDA.const, mPLDA.Icov, mPLDA.logDetCov, mPLDA.Icov2, mPLDA.logDetCov2 = mPLDA.prepare_scoring(5)

    return mPLDA