import torch
import numpy as np

# median  62.42532
# SIGMAS = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2,
#           1e-1, 1, 5, 10, 15, 20, 25,
#           30, 35, 100, 1e3, 1e4, 1e5, 1e6)


SIGMAS = [62.42532*x for x in np.exp2(np.arange(-8, 8+0.5, step=0.5))]


def mmd(x, y, sigmas=SIGMAS):
    '''
    
    :param x: matrix batch_size X feature_size
    :param y: matrix batch_size X feature_size
    :param sigmas: gaussian kernel bandwidths
    :return: mmd distant
    '''
    cost = gaussian_kernel(x, x, sigmas) \
          + gaussian_kernel(y, y, sigmas) \
          - 2 * gaussian_kernel(x, y, sigmas)
    return cost
    # return cost.sqrt()


def gaussian_kernel(x, y, sigmas):
    sigmas = torch.tensor(sigmas, device=x.get_device())
    beta = 1. / (2. * sigmas[:, None, None])
    dist = my_cdist(x, y)[:, None, None]
    # Todo do not know if it's ok
    # dist = dist / x.shape[-1]  # averge dim
    s = -beta * dist
    return s.exp().mean()  # probably not mean # average sample


def my_cdist(x1, x2):
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    res = torch.addmm(x2_norm.transpose(-2, -1), x1, x2.transpose(-2, -1), alpha=-2).add_(x1_norm)
    res = res.clamp_min_(1e-30).sqrt_()
    return res