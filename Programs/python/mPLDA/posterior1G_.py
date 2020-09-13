import numpy as np
def pdf1D(x, mu, R, detIsigma):
    temp = x - mu
    p = 0.39894228040 * detIsigma * np.exp(-0.5 * temp * R * temp)
    return p

def posterior1G(l, pi, mu, sigma):
    """
     Posterior of one 1-D Gaussian
     Note: mvnpdf is very slow. We can speedup computation by writing our own function
           where the det(sigma2) is pre-computed.
           Another speed up is to input the 2x2 precision matrix
    """

    K = len(pi)
    posty = np.ones(shape=(1, K), dtype=np.float64)
    wlh = np.zeros(shape=(1, K), dtype=np.float64)
    for r in range(K):
        wlh[0, r] = pi[r, 0] * pdf1D(l, mu[r, 0].flatten(), 1/(np.square(sigma[r, 0])), 1/sigma[r, 0])
    temp = np.sum(wlh)
    assert np.isnan(temp).any() == 0, 'Assertion in posteror_y1: sum(wlh) is NaN'
    for k in range(K):
        posty[0, k] = wlh[0, k] / temp
    return posty