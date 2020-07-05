# Use t-SNE to plot Kaldi's x-vectors

from kaldi_arkread import load_scpfile
import numpy as np
from sklearn.manifold import TSNE
from plot_ivectors import scatter2D
import matplotlib.pyplot as plt
from keras.models import load_model
from util_func import load_xvectors, select_speakers


def main():
    expdir = '../v2-1/exp/'     # Directory storing x-vectors from different domains
    datadir = '../v2-1/data/'   # Kaldi data dirctory storing info (e.g. spk2utt) of individual domains
    basedir = '/home5a/mwmak/so/spkver/sre18-dev/v2-1/'   # Base dir of Kaldi evaluation
    min_n_vecs = 20             # Min number of x-vectors per speaker 
    domains = ['swbd2', 'swbdcell', 'sre04-10-mx6', 'sre16', 'sitw', 'voxceleb1', 'sre18_dev_cmn2', 'sre18_dev_vast'] 
    #domains = ['sre18_dev_cmn2', 'sre16']
 
    # Load x-vectors from expdir/domains and in-domain
    X, spk_lbs, dom_lbs = load_xvectors(domains, expdir, datadir, basedir, min_n_vecs=min_n_vecs)
    X, spk_lbs, dom_lbs = select_speakers(X, spk_lbs, dom_lbs, min_n_vecs=20, n_spks=50)
    print('No. of x-vectors = %d' % int(X.shape[0]))
    print('No. of speakers = %d' % int(np.max(spk_lbs)+1))
    print('No. of domains = %d' % int(np.max(dom_lbs)+1))

    # Load ADSAN and transform x-vectors
    encoder = load_model('models/adnet_emb-400-20-epoch25.h5')
    X_enc = encoder.predict(X)

    print('Creating t-SNE plot of original x-vectors')
    scatter2D(TSNE(random_state=20150101).fit_transform(X), dom_lbs, domains)
    plt.show(block=False)

    print('Creating t-SNE plot of embeded x-vectors')
    scatter2D(TSNE(random_state=20150101).fit_transform(X_enc), dom_lbs, domains)
    plt.show(block=True)


if __name__ == '__main__':
    main()
