# Train a conventional DNN for speaker identifiction using raw x-vectors as input
#
# Speaker IDs can be extracted from Kaldi's spk2utt under the 'data/dataset' folder.
# For example, 'data/swbd/spk2utt' contains <spkid> <xvec-id1> <xvec-id2>...,
# where <xvec-idx> match the key field of exp/xvectors_swbd/xvector.scp.
#
# To run this script using Python3.6 (enmcomp3,4,11,13),
# assuming that Anaconda3 environment "tf-py3.6"
# has been created already
#   bash
#   export PATH=/usr/local/anaconda3/bin:/usr/local/cuda-8.0/bin:$PATH
#   export LD_LIBRARY_PATH=/usr/local/cuda-8.0/lib64:/usr/local/cuda-7.5/lib64
#   source activate tf-py3.6
#   python3 xvec_dnn_cls.py
#   source deactivate tf-py3.6

# M.W. Mak, Oct. 2018

from __future__ import print_function

import tensorflow as tf
from myPlot import scatter2D
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from keras.utils import np_utils
from DNN import DNN
from util_func import load_xvectors, split_trn_tst, select_speakers, get_accuracy


# Main function
def main():
    # Define some constants
    expdir = '../v2-1/exp/'     # Directory storing x-vectors from different domains
    vectype = 'xvector'         # Type of vectors, either 'ivector' or 'xvector'
    datadir = '../v2-1/data/'
    basedir = '/home2a/mwmak/so/spkver/sre18-dev/v2-1/'
    n_epochs = 25
    eint = 25                # Epoch interval for producing t-sne plots
    min_n_vecs = 2
    #out_dom = ['swbd2', 'swbdcell', 'sre04-10-mx6', 'voxceleb1', 'sre16']
    #out_dom = ['sre18_dev_cmn2', 'sre18_dev_vast']
    out_dom = ['sre04-10-mx6']
    #in_dom = ['sre18_dev_cmn2', 'sre18_dev_vast']
    in_dom = ['sre04-10-mx6']

    # Load x-vectors from expdir/domains and in-domain
    x_out, spk_lbs_out, dom_lbs_out, utt_ids_out = load_xvectors(out_dom, expdir,
                                                                 vectype, datadir, basedir,
                                                                 min_n_vecs=min_n_vecs)
    x_in, spk_lbs_in, dom_lbs_in, utt_ids_in = load_xvectors(in_dom, expdir,
                                                             vectype, datadir,
                                                             basedir, min_n_vecs=10)
    n_spk_out = int(np.max(spk_lbs_out)+1)
    n_dom_out = int(np.max(dom_lbs_out)+1)
    n_spk_in = int(np.max(spk_lbs_in)+1)
    n_dom_in = int(np.max(dom_lbs_in)+1)
    print('No. of x-vectors in out-domain = %d' % int(x_out.shape[0]))
    print('No. of speakers in out-domain = %d' % n_spk_out)
    print('No. of out domains = %d' % n_dom_out)
    print('No. of x-vectors in in-domain = %d' % int(x_in.shape[0]))
    print('No. of speakers in in-domain = %d' % n_spk_in)
    print('No. of in domains = %d' % n_dom_in)

    # Split out-domain data into training set and test set
    x_trn, spk_lbs_trn, _, _, x_tst, spk_lbs_tst, _, _ = split_trn_tst(
                                            x_out, spk_lbs_out, dom_lbs_out, utt_ids_out)

    # Convert to one-hot encoding
    spk_lbs_trn_1h = np_utils.to_categorical(spk_lbs_trn)
    spk_lbs_tst_1h = np_utils.to_categorical(spk_lbs_tst)

    # Select speakers for producing t-sne plots 
    x_sel, spk_lbs_sel, dom_lbs_sel, _ = select_speakers(x_in, spk_lbs_in, dom_lbs_in,
                                                         utt_ids_in, min_n_vecs=20, n_spks=10)

    # Plot original x-vectors on 2-D t-SNE space 
    print('Creating t-SNE plot of x-vectors')
    x_prj = TSNE(random_state=20150101).fit_transform(x_sel)
    fig, _, _, _ = scatter2D(x_prj, spk_lbs_sel, markers=dom_lbs_sel, n_colors=np.max(spk_lbs_sel)+1,
                             title='Original x-vectors')
    fig.savefig('logs/xvec.png')
    plt.show(block=False)

    # Create a DNN object
    net = DNN(f_dim=x_trn.shape[1], n_cls=n_spk_out, hnode=[512,512])

    # Iteratively train and test the DNN. 
    for it in range(int(n_epochs/eint)):
        epoch = (it + 1)*eint
        net.train(x_trn, spk_lbs_trn_1h, n_epochs=eint, batch_size=128)
        y_spk_pred = net.classifier.predict(x_tst)
        print('\nEpoch %d: Speaker ID Acc = %.2f%%' % (epoch, get_accuracy(y_spk_pred, spk_lbs_tst_1h)))
    
        # Save models to .h5 files
        net.classifier.save('models/dnn-epoch%d.h5' % epoch)


if __name__ == '__main__':
    # Use 1/3 of the GPU memory so that the GPU can be shared by multiple users
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    np.random.seed(1)

    # Ignore warnings
    def warn(*args, **kwargs):
        pass
    import warnings 
    warnings.warn = warn
    main()


