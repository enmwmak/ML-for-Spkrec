import numpy as np
def pdf2D(x, mu, R, detIsigma):
    temp = x - mu
    R = np.diag(R)
    p = 0.159154943 * detIsigma * np.exp(-0.5 * np.sum(temp * temp * R))
    return p

def posterior2G(ls, lt, pi, mu, sigma):
    """
     Posterior of two 1-D Gaussian
     Note: mvnpdf is very slow. We can speedup computation by writing our own function
           where the det(sigma2) is pre-computed.
           Another speed up is to input the 2x2 precision matrix
    """

    K = len(pi)
    wlh = np.zeros(shape=(K, K), dtype=np.float64)
    sigmaSq = sigma * sigma
    IsigmaSq = 1 / sigmaSq
    Isigma = 1 / sigma

    for p in range(K):
        for q in range(K):
            mu2 = np.vstack((mu[p, 0], mu[q, 0]))
            detIsigma = Isigma[p, 0] * Isigma[q, 0]
            preMat = np.diag(np.hstack((IsigmaSq[p, 0], IsigmaSq[q, 0])))
            wlh[p, q] = pi[p, 0] * pi[q, 0] * pdf2D(np.vstack((ls, lt)), mu2, preMat, detIsigma)

    temp = np.sum(np.sum(wlh))
    assert temp > 0, 'Divided by zeros in posterior_y2 of mPLDA_GroupScoring.m'
    posty = wlh / temp
    return posty