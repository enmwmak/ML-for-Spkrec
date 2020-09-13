"""
 Use Dataset 1 or Dataset 2 as defined in T-ASLP paper to train SImPLDA
 and SDmPLDA models (male)
 This script assumes that the following directories already exist, and it will
 write to these directories
   ./mat/fw60
   ./evl/fw60
   ./bosaris

The default setting of this script will produce the results in CC4 (male, Set I)
of Table III of the T-ASLP paper. Note that because the mPLDA model is initialized
randomly (see mPLDA_train5_.py), the results will be slightly different from run-to-run.
"""

import time
import numpy as np
from mPLDA_train4_ import mPLDA_train4
from mPLDA_train5_ import mPLDA_train5
from mPLDA_train3_ import mPLDA_train3
from scipy.io import loadmat, savemat
from comp_mGPLDA_ import comp_mGPLDA
from snr3_score_gplda_w_ import snr3_score_gplda_w
from evl2evl_ import evl2evl
from eval_by_bosaris_ import eval_by_bosaris

class Opt:

    def __init__(self):
        self.mode = []
        self.mtype = []

time_begian = time.time()

# Define constants
dataset = 1                              # Can be 1 or 2 (Set I and Set II in paper)
mtype = 'SDmPLDA'                        # Can be 'SImPLDA' or 'SDmPLDA'
opt = Opt()
opt.mode = 'scravg'
opt.mtype = mtype
nMixSet = [3]                              # No. of mixtures, e.g., nMixSet = [2, 3, 4]
cc = [4]                                   # SRE12 common conditions, e.g., cc = [4 5]
keyfile = 'key/NIST_SRE12_core_trial_key.v1'

# Determine which training algorithm to run
if mtype == 'SImPLDA':
    mPLDA_trainfunc = mPLDA_train5
elif mtype == 'SDmPLDA':
    mPLDA_trainfunc = mPLDA_train4
else:
    mPLDA_trainfunc = mPLDA_train3

# Load training i-vectors and SNR information
trndatafile = "./mat/fw60/male_target-dataset%d_mix_t500_w_1024c.mat" % dataset
print('Loading datafile %s\n' % trndatafile)
tgt = loadmat(trndatafile)

# Define target i-vector file and target SNR file for scoring
tgt_ivec_file = ['./mat/fw60/male_target-tel_mix_t500_w_1024c.mat', './mat/fw60/male_target-tel-15dB_mix_t500_w_1024c.mat', './mat/fw60/male_target-tel-06dB_mix_t500_w_1024c.mat']
tgt_snr_file = ['./snr/male/male_target-tel_stats.snr', './snr/male/male_target-tel-15dB_stats.snr', './snr/male/male_target-tel-06dB_stats.snr']

# Define test i-vec file and test snr file for scoring
tst_ivec_file = "./mat/fw60/male_test-tel-phn_mix_t500_w_1024c.mat"
tst_snr_file = "./snr/male/male_test-tel-phn_stats.snr"

# For each number of mixtures k, train and score an mPLDA model
for r in range(len(nMixSet)):
    k = nMixSet[r]

    # Define mPLDA and evaluation files
    mpldafile = './mat/fw60/male_mix_t500_%s_dataset%d-K%d_1024c.npy' % (mtype, dataset, k)
    print('Training %s\n' % mpldafile)
    GPLDAModel = comp_mGPLDA(tgt['w'], tgt['snr'], tgt['spk_logical'], mpldafile, 150, k, mPLDA_trainfunc)

    for j in range(len(cc)):
        evlfile = './evl/fw60/male_mix_t500_%s_dataset%d-K%d_1024c_cc%d.evl' % (mtype, dataset, k, cc[j])
        print('evlfile=%s\n' % evlfile)
        sre12_evlfile = './evl/fw60/sre12_male_mix_t500_%s_dataset%d-K%d_1024c_cc%d.evl' % (mtype, dataset, k, cc[j])
        res12_file = './bosaris/male_cleantest_%s_dataset%d-K%d_cc%d.npy' % (mtype, dataset, k, cc[j])
        ndx_lstfile = './lst/fw60/male_ndx_cc%d_stats.lst' % cc[j]
        ndxfile = './ndx/male/core-core_8k_male_cc%d.ndx' % cc[j]

        snr3_score_gplda_w(tgt_ivec_file, '', tst_ivec_file, ndx_lstfile, '', '', 'None', evlfile,
                           mpldafile, tst_snr_file, tgt_snr_file, opt)
        evl2evl(evlfile, ndxfile, sre12_evlfile)

        ccres = np.empty(shape=(k, j+1), dtype=object)
        ccres[k-1, j] = {}
        eer, dcf12, res12 = eval_by_bosaris(sre12_evlfile, keyfile, fast_keyloading = False)
        ccres[k - 1, j].update({'eer': eer})
        ccres[k - 1, j].update({'dcf12': dcf12})
        ccres[k - 1, j].update({'res12': res12})

        eer = ccres[k - 1, j]['eer']
        dcf12 = ccres[k - 1, j]['dcf12']
        res12 = ccres[k-1, j]['res12']
        print('Saving result to %s\n' % res12_file)
        np.save(res12_file, ccres[k-1, j])  # save results (a dict) to .npy file

time_end = time.time()
print("Total time consume: %f seconds" % (time_end - time_begian))
