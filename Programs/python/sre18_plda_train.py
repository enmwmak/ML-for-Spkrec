"""
Train PLDA model for SRE18-dev and SRE18-eval using Kaldi's i-vectors
Note that for training, we could directly read the .ark files in another order
as Kaldi_Plda can use the spk_ids as the keys. However, for scoring, we must
use .scp files to load the i-vectors as batch_plda_scoring.py requires the
i-vectors to be in order for proper scoring.
"""

from plda.kaldi_plda import Kaldi_Plda
from kaldi_arkread import scp2h5, arkdir2h5
import os
import sys

# Define constants and options
__scp2h5 = False            # Very slow because reading one vector at a time
__ark2h5 = True             # Much faster but i-vecs will be out-of-order
vtype = 'ivector'           # Type of vectors ('ivector'|'xvector')
arktype = 'vec'             # ark files store one vector per key

if vtype == 'ivector':
    basedir = '/home2a/mwmak/so/spkver/sre18-dev/v1-1/'
else:
    basedir = '/home2a/mwmak/so/spkver/sre18-dev/v2-1/'
scpfiles = [
    basedir + 'exp/%ss_sre_combined/%s.scp' % (vtype, vtype),
    basedir + 'exp/%ss_sre18_dev_unlabeled/%s.scp' % (vtype, vtype)
]
arkdirs = [
    basedir + 'exp/%ss_sre_combined' % vtype,
    basedir + 'exp/%ss_sre18_dev_unlabeled' % vtype,
]
h5dir = 'data/h5/'
h5files = [
    h5dir + '%ss_sre_combined.h5' % vtype,
    h5dir + '%ss_sre18_dev_unlabeled.h5' % vtype,
]

############################
# train a PLDA model
############################
if len(sys.argv) != 2:
    print('Usage: %s <prep_mode>' % sys.argv[0])
    print('       Valid prep_mode: lda+lennorm')
    exit()

prep_mode = sys.argv[1]

# Read i-vectors from .ark dir and save them as one .h5 file
# Kaldi add '_<dataset>_<segment-name>' or '-<dataset>_<segment-name>'
# to the key part of the .ark file.
# It is important to strip the dataset and segment-name; otherwise,
# each speaker will be considered as providing ONE utterance only.
if __scp2h5:
    for sfile, hfile in zip(scpfiles, h5files):
        os.remove(hfile) if os.path.isfile(hfile) else None
        print('{0} --> {1}'.format(sfile, hfile))
    scp2h5(basedir, scpfiles[0], h5files[0], strip_after_chars=['-', '_'])
    scp2h5(basedir, scpfiles[1], h5files[1], strip_after_chars=None)

if __ark2h5:
    for adir, hfile in zip(arkdirs, h5files):
        os.remove(hfile) if os.path.isfile(hfile) else None
        print('{0} --> {1}'.format(adir, hfile))
        pattern = '%s.*.ark' % vtype
        arkdir2h5(adir, hfile, pattern=pattern, strip_after_chars=['-', '_'],
                  arktype=arktype)

# Train model
indomain_datafiles = [h5dir + '%ss_sre18_dev_unlabeled.h5' % vtype]
model_file = h5dir + 'model/kaldi-%s-plda-sre18-%s.h5' % (vtype, prep_mode)
plda = Kaldi_Plda(
    n_iter=10,
    n_fac=300,
    lda_dim=300,                        # Valid for prep_mode='wccn+lennorm+lda+wccn' only
    datafiles=[h5dir + '%ss_sre_combined.h5' % vtype],
    prep_mode=prep_mode,
    indomain_datafiles=indomain_datafiles,
    model_file=model_file
)
plda.fit()

