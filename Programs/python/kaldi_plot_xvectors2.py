# Use t-SNE to plot Kaldi's x-vectors

from kaldi_arkread import load_scpfile
import numpy as np
from sklearn.manifold import TSNE
from plot_ivectors import scatter2D
import matplotlib.pyplot as plt
from keras.models import load_model


def main():
    basedir = '/home5a/mwmak/so/spkver/sre18-dev/v2-1/'
    scpfile = [
        basedir + 'exp/xvectors_voxceleb1/xvector.scp',
        basedir + 'exp/xvectors_sitw/xvector.scp',
        basedir + 'exp/xvectors_swbd2/xvector.scp',
        basedir + 'exp/xvectors_swbdcell/xvector.scp',
        basedir + 'exp/xvectors_sre04-10-mx6/xvector.scp',
        basedir + 'exp/xvectors_sre16/xvector.scp',
        basedir + 'exp/xvectors_sre18_eval_test/xvector.scp'       
    ]
    domains = ['voxceleb1','sitw','swbd2', 'swbdcell', 'sre04-10-mx6', 'sre16','sre18_eval_test']
    max_smps = 1500         # Maximum no. of vectors per domain

    X = list()
    Y = list()
    for i in range(len(scpfile)):
        _, mat = load_scpfile(basedir, scpfile[i], arktype='vec')
        mat = mat[0:np.min([mat.shape[0], max_smps]), :]
        X.append(mat)
        Y.extend([i] * mat.shape[0])
    X = np.vstack(X)
    Y = np.array(Y)

    # Load ADSAN and transform x-vectors
    #encoder = load_model('models/adnet_emb-400-20-epoch25.h5')
    #X_enc = encoder.predict(X)

    print('Creating t-SNE plot of original x-vectors')
    fig, _, _ = scatter2D(TSNE(random_state=20150101).fit_transform(X), Y, domains)
    fig.savefig('logs/multi-dataset-xvec.png')
    #plt.show(block=False)
    plt.show(block=True)

    #print('Creating t-SNE plot of embeded x-vectors')
    #fig, _, _ = scatter2D(TSNE(random_state=20150101).fit_transform(X_enc), Y, domains)
    #fig.savefig('logs/multi-dataset-emb-xvec.png')
    #plt.show(block=True)


if __name__ == '__main__':
    main()
