# Use t-SNE to plot Kaldi's i-vectors

from kaldi_arkread import load_scpfile
import numpy as np
from sklearn.manifold import TSNE
from plot_ivectors import scatter2D
import matplotlib.pyplot as plt


def main():
    basedir = '/home5a/mwmak/so/spkver/sre18-dev/v1-1/'
    scpfile = [
        basedir + 'exp/ivectors_sre18_dev_unlabeled/ivector.scp',
        basedir + 'exp/ivectors_sre18_dev_pstn/ivector.scp',
        basedir + 'exp/ivectors_sre18_dev_voip/ivector.scp',
        basedir + 'exp/ivectors_sre18_dev_vast/ivector.scp',
        basedir + 'exp/ivectors_voxceleb1/ivector.scp',
        basedir + 'exp/ivectors_sitw_eval_test/ivector.scp'
    ]
    sources = ['sre18_unlabeled', 'sre18_pstn', 'sre18_voip', 'sre18_vast',
               'voxceleb1', 'sitw']
    max_smps = 1500

    X = list()
    Y = list()
    for i in range(len(scpfile)):
        _, mat = load_scpfile(basedir, scpfile[i], arktype='vec')
        mat = mat[0:np.min([mat.shape[0], max_smps]), :]
        X.append(mat)
        Y.extend([i] * mat.shape[0])
    X = np.vstack(X)
    Y = np.array(Y)

    print('Creating t-SNE plot')
    X_emb = TSNE(random_state=20150101).fit_transform(X)
    scatter2D(X_emb, Y, sources)
    plt.show()


if __name__ == '__main__':
    main()
